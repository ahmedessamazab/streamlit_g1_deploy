import streamlit as st
import os
from huggingface_hub import InferenceClient

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

st.title("Hello, Streamlit!")

sentence = st.text_input("Enter your sentence:")


client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

if sentence:
    result = client.text_classification(
        sentence,
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )

    st.write(f"Classification Result: {result[0]['label']}")
