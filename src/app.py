import openai
import streamlit as st
import pdf_gpt
from pinecone import Pinecone, ServerlessSpec
import os
from io import BytesIO

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY_V2"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

api_key = st.secrets["OPENAI_API_KEY"]
api_base = st.secrets["OPENAI_API_BASE"]
api_version = st.secrets["OPENAI_API_VERSION"]
api_type = st.secrets["OPENAI_API_TYPE"]

openai.api_key = api_key
openai.api_base = api_base
openai.api_version = api_version
openai.api_type = api_type

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_EMBEDDING_ENGINE = 'mondongodb'
DIMENSION = 1536
GPT_MODEL = 'gpt-3.5-turbo-16k'
GPT_CHAT_ENGINE = "gepeto"

st.title("Chatbot")
is_pdf_chatbot = st.checkbox("PDF chatbot")
uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hola, soy ChatGPT, ¿En qué puedo ayudarte?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    pdf_file = BytesIO(pdf_bytes)
    # Sube el archivo a Pinecone
    pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.index("fp-incompany-index")
    index.upsert(ids=[uploaded_file.name], embeddings=[pdf_bytes])
    st.success("¡Archivo PDF subido a Pinecone exitosamente!")
    pdf_file.close()

if user_input := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Recupera el archivo desde Pinecone
    embeddings = index.query(queries=[user_input], top_k=1)
    if embeddings:
        pdf_bytes = embeddings[0]["embedding"]
        # Procesa el archivo PDF y realiza la conversación con OpenAI
        docsearch = pdf_gpt.process_pdf(BytesIO(pdf_bytes), api_key, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, DIMENSION)
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=st.session_state["messages"],
            engine=GPT_CHAT_ENGINE,
            max_tokens=DIMENSION
        )
        responseMessage = response['choices'][0]['message']['content']
        st.session_state["messages"].append({"role": "assistant", "content": responseMessage})
        st.chat_message("assistant").write(responseMessage)
    else:
        st.warning("No se encontraron documentos relacionados en Pinecone.")
