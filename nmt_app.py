import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pytesseract
from PIL import Image

# Load model and tokenizers
model = load_model('files/multitask_nmt_final.h5', compile=False)

eng_tokenizer = pickle.load(open('files/eng_tokenizer.pkl', 'rb'))
sinhala_tokenizer = pickle.load(open('files/sinhala_tokenizer.pkl', 'rb'))
singlish_tokenizer = pickle.load(open('files/singlish_tokenizer.pkl', 'rb'))

eng_max_length = 47  
sinhala_max_length = 50
singlish_max_length = 50

# Helper functions
def encode_sequences(tokenizer, max_length, lines):
    X = tokenizer.texts_to_sequences(lines)
    return pad_sequences(X, maxlen=max_length, padding='post')

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_sequence_with_confidence(model, source, target_tokenizer, output_name):
    prediction = model.predict(source, verbose=0)[0 if output_name == 'sinhala_output' else 1][0]
    integers = [np.argmax(vector) for vector in prediction]
    probs = [np.max(vector) for vector in prediction]
    words = [word_for_id(i, target_tokenizer) for i in integers if word_for_id(i, target_tokenizer)]
    confidence = np.mean(probs)
    return ' '.join(words), round(confidence * 100, 2)

# Streamlit UI
st.markdown('<h1 style="color:orange;">English to Sinhala & Singlish Translator with Confidence</h1>', unsafe_allow_html=True)
st.markdown('<h4>Choose how you want to input English text:</h4>', unsafe_allow_html=True)

# Input selection
option = st.radio("Input Method", ("Type Text", "Upload .txt File"))
input_text = ""

if option == "Type Text":
    input_text = st.text_input("Enter English sentence:")

elif option == "Upload .txt File":
    uploaded_txt = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_txt is not None:
        input_text = uploaded_txt.read().decode("utf-8")
        st.success(f"Loaded Text: {input_text.strip()}")

# Translation Section
if st.button("Translate"):
    if input_text.strip():
        encoded_input = encode_sequences(eng_tokenizer, eng_max_length, [input_text.lower()])
        sinhala_translation, sin_conf = predict_sequence_with_confidence(model, encoded_input, sinhala_tokenizer, 'sinhala_output')
        singlish_translation, sing_conf = predict_sequence_with_confidence(model, encoded_input, singlish_tokenizer, 'singlish_output')

        st.markdown('<h3 style="color:green;">Sinhala Translation:</h3>', unsafe_allow_html=True)
        st.write(sinhala_translation)
        st.markdown(f'<span style="color:yellow;">Confidence:</span> <span style="color:white; font-weight:bold;">{sin_conf:.2f}%</span>', unsafe_allow_html=True)

        st.markdown('<h3 style="color:green;">Singlish Translation:</h3>', unsafe_allow_html=True)
        st.write(singlish_translation)
        st.markdown(f'<span style="color:yellow;">Confidence:</span> <span style="color:white; font-weight:bold;">{sing_conf:.2f}%</span>', unsafe_allow_html=True)
    else:
        st.warning("Please enter or upload a sentence to translate.")