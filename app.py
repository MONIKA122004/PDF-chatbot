import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
import json

# Title
st.set_page_config(page_title="📄 PDF Chatbot")
st.title("📄 Chat with your PDF")

# Upload PDF
pdf = st.file_uploader("Upload your PDF file", type="pdf")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load API key
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if pdf and groq_api_key:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split into chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Display previous chat
    for chat in st.session_state.chat_history:
        st.chat_message(chat["role"]).markdown(chat["content"])

    # User input
    prompt = st.chat_input("Ask something about the PDF...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Call Groq
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Replace with available model
            messages=st.session_state.chat_history
        )

        bot_msg = response.choices[0].message.content
        st.chat_message("assistant").markdown(bot_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_msg})

        # Save chat history to a file (optional)
        with open("chat_history.json", "w") as f:
            json.dump(st.session_state.chat_history, f, indent=4)

elif not groq_api_key:
    st.warning("Please set your GROQ_API_KEY in .env or Streamlit secrets.")
