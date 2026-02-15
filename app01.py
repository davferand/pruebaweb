import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuración de la página
st.set_page_config(page_title="IA Business & Support Tool", layout="wide")

# --- BARRA LATERAL: Configuración ---
st.sidebar.title("Configuración")
api_key = st.sidebar.text_input("Introduce tu Groq API Key:", type="password")

if not api_key:
    st.info("Por favor, introduce tu API Key de Groq en la barra lateral para comenzar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# --- CARGA DE MODELO LLM ---
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

llm = get_llm()

# --- LÓGICA RAG (Atención al Cliente) ---
@st.cache_resource
def prepare_rag_system(file_path):
    # 1. Cargar PDF
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Dividir texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    # 3. Crear Embeddings e Indexar en FAISS
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# --- INTERFAZ PRINCIPAL ---
st.title("🤖 IA para Marketing y Atención al Cliente")
tab1, tab2 = st.tabs(["📢 Generador de Marketing", "🎧 Atención al Cliente (RAG)"])

# --- TAB 1: GENERADOR DE MARKETING ---
with tab1:
    st.header("Generación de Contenido SEO")
    
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Tema del contenido:", placeholder="Ej: Nutrición deportiva")
        platform = st.selectbox("Plataforma:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'Email'])
        tone = st.selectbox("Tono:", ['Normal', 'Informativo', 'Inspirador', 'Urgente', 'Informal'])
    
    with col2:
        length = st.selectbox("Longitud:", ['Short', 'Medium', 'Long', '1 paragraph'])
        audience = st.selectbox("Audiencia Objetivo:", ['All', 'Young adults', 'Families', 'Seniors', 'Teenagers'])
        keywords = st.text_area("Keywords (SEO):", placeholder="Ej: fitness, salud, dieta...")

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        cta = st.checkbox("Incluir Call to Action (CTA)")
    with c_col2:
        hashtags = st.checkbox("Incluir Hashtags")

    if st.button("Generar Contenido de Marketing"):
        if not topic:
            st.warning("Por favor, introduce un tema.")
        else:
            marketing_prompt = f"""
            Write an SEO-optimized text on the topic '{topic}'.
            Return only the final text in your response and don't put it inside quotes.
            - Platform where it will be published: {platform}.
            - Tone: {tone}.
            - Target audience: {audience}.
            - Length: {length}.
            - {"Include a clear Call to Action." if cta else "Do not include a Call to Action."}
            - {"Include relevant hashtags at the end of the text." if hashtags else "Do not include hashtags."}
            {"- Keywords to include (for SEO): " + keywords if keywords else ""}
            """
            
            template = ChatPromptTemplate.from_messages([
                ("system", "You are a digital marketing expert specialized in SEO and persuasive copywriting."),
                ("human", "{prompt}")
            ])
            
            chain = template | llm | StrOutputParser()
            
            with st.spinner("Generando..."):
                result = chain.invoke({"prompt": marketing_prompt})
                st.markdown("### Resultado:")
                st.write(result)

# --- TAB 2: ATENCIÓN AL CLIENTE (RAG) ---
with tab2:
    st.header("Asistente Virtual SafeBank")
    st.write("Haz preguntas sobre el manual de SafeBank.")

    # El archivo PDF debe estar en la misma carpeta que app.py en GitHub
    pdf_path = "safebank-manual.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"Archivo {pdf_path} no encontrado. Asegúrate de subirlo a tu repositorio.")
    else:
        retriever = prepare_rag_system(pdf_path)
        
        user_question = st.text_input("Escribe tu duda aquí:")
        
        if user_question:
            system_prompt = """You are a helpful virtual assistant answering general questions about a company's services.
            Use the following bits of retrieved context to answer the question.
            If you don't know the answer, just say you don't know. Keep your answer concise. \n\n Context: {context}"""

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Question: {input}"),
            ])

            rag_chain = (
                {"context": retriever, "input": RunnablePassthrough()}
                | qa_prompt
                | llm
                | StrOutputParser()
            )

            with st.spinner("Buscando en el manual..."):
                response = rag_chain.invoke(user_question)
                st.markdown("### Respuesta del Asistente:")
                st.write(response)
