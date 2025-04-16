import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from torch.utils.data import Dataset
import torch
import numpy as np
import evaluate
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration - CPU specific
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "kan_Knda"  # Kannada
TGT_LANG = "sdd_Sidd"  # Siddi
MAX_LENGTH = 128
BATCH_SIZE = 4  # Reduced for CPU
NUM_EPOCHS = 5  # Reduced for CPU
LEARNING_RATE = 3e-5  # Adjusted for CPU
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "kannada_siddi_translator_cpu"
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

logger.info("Loading tokenizer and model for CPU...")
# Force CPU usage
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# Special tokens for language pairs
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

logger.info(f"Model loaded on {device}: {MODEL_NAME}")

# Fixed Custom dataset class
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_length=128):
        logger.info(f"Loading dataset from {src_file} and {tgt_file}")
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_texts = [line.strip() for line in f if line.strip()]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_texts = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loaded {len(self.src_texts)} parallel sentences")

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = str(self.src_texts[idx])
        tgt_text = str(self.tgt_texts[idx])

        # Tokenize without return_tensors="pt"
        model_inputs = tokenizer(
            src_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )

        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# Load datasets
logger.info("Loading training and validation datasets...")
train_dataset = TranslationDataset(
    "dataset/src-train.txt",
    "dataset/tgt-train.txt",
    tokenizer,
    MAX_LENGTH
)

val_dataset = TranslationDataset(
    "dataset/src-val.txt",
    "dataset/tgt-val.txt",
    tokenizer,
    MAX_LENGTH
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8  # Helps with CPU efficiency
)

# Metrics
try:
    metric = evaluate.load("sacrebleu")
except Exception as e:
    logger.warning(f"Couldn't load sacrebleu: {e}")
    metric = None

def compute_metrics(eval_preds):
    if metric is None:
        return {}

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# CPU-optimized training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=2,  # Reduced for CPU
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    fp16=False,  # Disabled for CPU
    bf16=False,  # Disabled for CPU
    logging_dir=LOGGING_DIR,
    logging_steps=50,  # More frequent logging
    save_strategy="epoch",
    report_to="none",  # Disabled wandb for CPU
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="bleu" if metric else None,
    greater_is_better=True,
    no_cuda=True,  # Force CPU
    dataloader_pin_memory=False,  # Disabled for CPU
    disable_tqdm=False,
    logging_first_step=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics if metric else None
)

# Training
logger.info("Starting CPU training...")
try:
    train_result = trainer.train()
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# Save model
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
logger.info(f"Model saved to {os.path.join(OUTPUT_DIR, 'final_model')}")

# Test evaluation
test_dataset = TranslationDataset(
    "dataset/src-test.txt",
    "dataset/tgt-test.txt",
    tokenizer,
    MAX_LENGTH
)

test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
logger.info(f"Test results: {test_results}")

# Save predictions
predictions = trainer.predict(test_dataset)
preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

with open(os.path.join(OUTPUT_DIR, "predictions.txt"), 'w', encoding='utf-8') as f:
    for src, pred, tgt in zip(test_dataset.src_texts, preds, test_dataset.tgt_texts):
        f.write(f"SRC: {src}\nPRED: {pred}\nTGT: {tgt}\n\n")

logger.info("Training complete!")