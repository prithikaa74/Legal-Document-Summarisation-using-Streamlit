import numpy as np
import PyPDF2
from transformers import BartTokenizer, BartForConditionalGeneration
from googletrans import Translator
from rouge_score import rouge_scorer
import torch

# Define PDF extraction function
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Preprocess the text
def preprocess_text(text):
    # Remove unnecessary newlines and split into sentences
    sentences = [sent.strip() for sent in text.split('\n') if sent.strip()]
    cleaned_text = ' '.join(sentences)  # Join sentences to form a clean paragraph
    return cleaned_text

# Load BART model and tokenizer with GPU acceleration if available
def load_model_and_tokenizer():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Check for GPU and move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Using GPU for summarization.")
    else:
        print("Using CPU for summarization.")
    return tokenizer, model

# Generate summary with approximate word count of 1000
def generate_summary(model, tokenizer, input_text, max_length=1333, beam_size=2):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Move input tensors to GPU if available
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
    
    # Generate the summary with smaller beam size for faster results
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=beam_size, early_stopping=True)
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Limit the summary to 1000 words
    summary_words = summary.split()
    summary = ' '.join(summary_words[:1000])  # Limit to 1000 words
    return summary

# Translate text using Google Translate API
def translate_text(text, target_language="en"):  # Default translation to English
    translator = Translator()

    # Define language map
    lang_map = {
        'tamil': 'ta',
        'telugu': 'te',
        'malayalam': 'ml',
        'kannada': 'kn',
        'english': 'en'
    }
    
    # Check if the selected language is valid
    if target_language.lower() in lang_map:
        translated = translator.translate(text, dest=lang_map[target_language.lower()])
        return translated.text
    else:
        return "Unsupported language! Please choose from Tamil, Telugu, Malayalam, Kannada, or English."

# Compute ROUGE scores
def compute_rouge_scores(reference_text, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_summary)
    return scores

# Main function to run the summarization, translation, and evaluation
def main():
    pdf_file_path = 'free-consultancy-terms-and-conditions.pdf'  # Update this path to your PDF file

    # Ask user for preferred language (Tamil, Telugu, Malayalam, Kannada, English)
    target_language = input("Choose the target language (Tamil, Telugu, Malayalam, Kannada, English): ").lower()

    full_text = extract_text_from_pdf(pdf_file_path)
    
    print("\nExtracted Text (first 1000 chars):")
    print(full_text[:1000])  # Display first 1000 characters of the PDF text

    # Preprocess text
    cleaned_text = preprocess_text(full_text)
    
    # Load model and tokenizer
    print("\nLoading the BART model...")
    tokenizer, model = load_model_and_tokenizer()

    # Generate and display the summary
    print(f"\nGenerating summary of approximately 1000 words...")
    summary = generate_summary(model, tokenizer, cleaned_text, beam_size=2)  # Set beam_size=2 for faster results

    print("\nGenerated Summary (1000 words):")
    print(summary)

    # Translate the summary to the target language
    print(f"\nTranslating the summary to {target_language.title()}...")
    translated_summary = translate_text(summary, target_language=target_language)

    print(f"\nTranslated Summary ({target_language.title()}):")
    print(translated_summary)

    # Compute ROUGE scores
    print("\nCalculating ROUGE scores for the generated summary...")
    rouge_scores = compute_rouge_scores(cleaned_text, summary)

    # Display ROUGE scores
    print("\nROUGE Scores:")
    print(f"ROUGE-1: {rouge_scores['rouge1']}")
    print(f"ROUGE-2: {rouge_scores['rouge2']}")
    print(f"ROUGE-L: {rouge_scores['rougeL']}")

if __name__ == '__main__':
    main()
