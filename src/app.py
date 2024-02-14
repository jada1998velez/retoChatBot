import openai
import streamlit as st
import pdf_gpt
from pinecone import Pinecone, ServerlessSpec
import os
from io import BytesIO
from transformers import BertTokenizer, BertModel


# Cargar modelo BERT preentrenado y tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Función para procesar el PDF
def process_pdf(file):
    pdf_document = PyMuPDF.PdfReader(file)
    pdf_text = [page.extract_text() for page in pdf_document.pages]
    
    paragraph_vectors = []
    for paragraph in pdf_text:
        tokens = tokenizer(paragraph, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**tokens)
        paragraph_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        paragraph_vectors.append(paragraph_vector)

    return paragraph_vectors

# Configuración de la aplicación Streamlit
st.title("PDF to Pinecone")

uploaded_file = st.file_uploader("Cargar archivo PDF", type=["pdf"])

if uploaded_file is not None:
    st.write("Archivo cargado con éxito!")

    # Procesar el PDF
    vectors = process_pdf(uploaded_file)

    # Aquí podrías enviar `vectors` a Pinecone o realizar otras operaciones según tus necesidades

    st.write("Vectores generados:")
    st.write(vectors)








