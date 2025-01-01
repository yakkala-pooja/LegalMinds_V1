from transformers import BartForConditionalGeneration, BartTokenizer

# Load pre-trained model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Function for generating summary
def generate_summary(text, max_length=500, min_length=100):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=min_length, max_length=max_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary