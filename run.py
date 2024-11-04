import streamlit as st
import PyPDF2
from transformers import BartTokenizer, BartForConditionalGeneration
from mtranslate import translate
from sentence_transformers import SentenceTransformer, util
import sqlite3
import torch
import logging
import hashlib

# Database initialization
def init_db():
    conn = sqlite3.connect("summarized_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            original_text TEXT,
            summary TEXT,
            translated_summary TEXT,
            accuracy REAL
        )
    ''')
    conn.commit()
    return conn, cursor

# Hash password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Register a new user
def register_user(username, password):
    conn, cursor = init_db()
    try:
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Authenticate an existing user
def authenticate(username, password):
    conn, cursor = init_db()
    cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

# Extract text from PDF
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

# Load BART model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    if torch.cuda.is_available():
        model = model.to('cuda')
    return tokenizer, model

# Load Sentence-BERT model
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Generate summary
def generate_summary(model, tokenizer, input_text, max_length=1333, beam_size=2):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {key: value.to('cuda') for key, value in inputs.items()}

    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=beam_size, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary_words = summary.split()
    return ' '.join(summary_words[::])

# Translate text
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

# Calculate accuracy with Sentence-BERT
def calculate_accuracy_with_sbert(original_text, generated_summary):
    sbert_model = load_sbert_model()

    # Encode original text and summary
    original_embedding = sbert_model.encode(original_text, convert_to_tensor=True)
    summary_embedding = sbert_model.encode(generated_summary, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_sim = util.cos_sim(original_embedding, summary_embedding).item()

    return cosine_sim * 100

# Store summary data in SQLite database
def save_summary_to_db(user, original_text, summary, translated_summary, accuracy):
    conn, cursor = init_db()
    cursor.execute('''
        INSERT INTO summaries (user, original_text, summary, translated_summary, accuracy) 
        VALUES (?, ?, ?, ?, ?)
    ''', (user, original_text, summary, translated_summary, accuracy))
    conn.commit()
    conn.close()

# Retrieve summaries for the logged-in user
def get_user_summaries(user):
    conn, cursor = init_db()
    cursor.execute('SELECT id, original_text, summary, translated_summary, accuracy FROM summaries WHERE user = ?', (user,))
    summaries = cursor.fetchall()
    conn.close()
    return summaries

# Streamlit UI with signup, login, and retrieval features
def main():
    st.title("Legal Document Summarization")
    st.sidebar.header("Account")

    # Toggle for Login or Signup
    auth_option = st.sidebar.selectbox("Choose Action", ["Login", "Signup"])

    # Registration Section
    if auth_option == "Signup":
        st.sidebar.subheader("Create a New Account")
        new_username = st.sidebar.text_input("Choose a Username")
        new_password = st.sidebar.text_input("Choose a Password", type="password")
        if st.sidebar.button("Signup"):
            if register_user(new_username, new_password):
                st.sidebar.success("Account created successfully! You can now log in.")
            else:
                st.sidebar.error("Username already exists. Please choose another.")

    # Login Section
    elif auth_option == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if authenticate(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("Login successful!")
            else:
                st.sidebar.error("Invalid username or password")

    # Check if authenticated
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        st.sidebar.header("Translation")

        # File upload and summarization
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

                # Calculate accuracy using Sentence-BERT
                accuracy = calculate_accuracy_with_sbert(cleaned_text, summary)
                st.write(f"Accuracy: {accuracy:.2f}%")

                # Save the data to the database
                save_summary_to_db(username, cleaned_text, summary, translated_summary, accuracy)
                st.success("Summary saved to database.")
            else:
                st.error("Please upload a PDF file.")

        # Display user summaries
        st.sidebar.subheader("View Saved Summaries")
        if st.sidebar.button("Show My Summaries"):
            summaries = get_user_summaries(st.session_state["username"])
            if summaries:
                st.subheader("Your Saved Summaries")
                for idx, (id, original_text, summary, translated_summary, accuracy) in enumerate(summaries, 1):
                    st.write(f"### Summary {idx}")
                    st.write(f"**Original Text:** {original_text[:500]}...")  # Truncate for display
                    st.write(f"**Summary:** {summary}")
                    st.write(f"**Translated Summary:** {translated_summary}")
                    st.write(f"**Accuracy:** {accuracy:.2f}%")
                    st.write("---")
            else:
                st.info("No saved summaries found.")
    else:
        st.warning("Please log in or sign up to access the summarization feature.")

if __name__ == "__main__":
    main()
