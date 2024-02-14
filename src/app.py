import streamlit as st
import fitz  # PyMuPDF

def leer_pdf(archivo_pdf):
    doc = fitz.open(archivo_pdf)
    num_paginas = doc.page_count

    st.sidebar.title("Seleccionar Página")
    pagina_seleccionada = st.sidebar.slider("Selecciona una página", 1, num_paginas, 1)

    pagina = doc.load_page(pagina_seleccionada - 1)
    texto = pagina.get_text("text")

    st.title(f"Página {pagina_seleccionada}")
    st.write(texto)

    doc.close()

def main():
    st.title("Lector de PDF en Streamlit")

    uploaded_file = st.file_uploader("Subir un archivo PDF", type=["pdf"])

    if uploaded_file is not None:
        st.success("Archivo cargado correctamente.")
        leer_pdf(uploaded_file)

if __name__ == "__main__":
    main()
