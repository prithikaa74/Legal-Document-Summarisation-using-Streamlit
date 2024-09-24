import streamlit as st
import PyPDF2
from transformers import BartTokenizer, BartForConditionalGeneration
from mtranslate import translate
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch
import logging

# Define PDF extraction function
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)  
    for page in reader.pages:
        text += page.extract_text() or "" 
    return text

# Preprocess the text
def preprocess_text(text):
    sentences = [sent.strip() for sent in text.split('\n') if sent.strip()]
    cleaned_text = ' '.join(sentences)
    return cleaned_text

# Load model
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    if torch.cuda.is_available():
        model = model.to('cuda')
    return tokenizer, model

# Generate summary
def generate_summary(model, tokenizer, input_text, max_length=1333, beam_size=2):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
    
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=beam_size, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary_words = summary.split()
    return ' '.join(summary_words[:1000])

# Translate 
def translate_text(text, target_language="en"):
    lang_map = {
        'tamil': 'ta',
        'telugu': 'te',
        'malayalam': 'ml',
        'kannada': 'kn',
        'english': 'en'
    }

    try:
        if target_language.lower() in lang_map:
            translated = translate(text, lang_map[target_language.lower()])
            return translated
        else:
            return "Unsupported language!"
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return "Translation not available at the moment."

# Calculate accuracy score 
def calculate_accuracy(original_text, generated_summary):
    vectorizer = CountVectorizer().fit_transform([original_text, generated_summary])
    vectors = vectorizer.toarray()

    # Convert vectors to float
    vectors = vectors.astype(np.float32)
    
    # Convert to PyTorch tensors
    vec1 = torch.tensor(vectors[0])
    vec2 = torch.tensor(vectors[1])
    
    # Calculate the cosine similarity between original and summary as a proxy for accuracy
    cosine_sim = (vec1 @ vec2) / (torch.norm(vec1) * torch.norm(vec2))
    
    return cosine_sim.item() * 100

# Streamlit U
def main():
    st.title("Legal Document Summarization")
    st.sidebar.header("Translation")
    
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    target_language = st.sidebar.selectbox("Choose Language", ["English", "Tamil", "Telugu", "Malayalam", "Kannada"])
    
    if st.button("Summarize"):
        if pdf_file is not None:
            full_text = extract_text_from_pdf(pdf_file)
            cleaned_text = preprocess_text(full_text)
            
            st.write("Loading the model...")
            tokenizer, model = load_model_and_tokenizer()
            
            st.write("Generating summary...")
            summary = generate_summary(model, tokenizer, cleaned_text)
            st.subheader("**Generated Summary:**")
            st.write(summary)
            
            translated_summary = translate_text(summary, target_language.lower())
            st.subheader(f"**Translated Summary ({target_language}):**")
            st.write(translated_summary)
            
            accuracy = calculate_accuracy(cleaned_text, summary)
            #st.subheader("Accuracy Score:")
            st.write(f"Accuracy: {accuracy:.2f}%")
        else:
            st.error("Please upload a PDF file.")

if __name__ == "__main__":
    main()
