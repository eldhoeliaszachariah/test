from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "kannada_siddi_translator/final_model"
SRC_LANG = "kan_Knda"
TGT_LANG = "sdd_Sidd"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

def translate(text):
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG],
        max_length=200
    )
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Example usage
kannada_text = "ನಿನಗೆ ಹೇಗೆ ಇದೆ?"  # "How are you?" in Kannada
siddi_translation = translate(kannada_text)
print(f"Kannada: {kannada_text}")
print(f"Siddi: {siddi_translation}")