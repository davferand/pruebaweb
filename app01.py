import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Generador de Contenido 🤖", page_icon="🤖")

## Connection with the LLM
id_model = "llama-3.3-70b-versatile"

def get_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model=id_model,
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

@st.cache_resource
def initialize_rag():
    file_path = "safebank-manual.pdf"
    if not os.path.exists(file_path):
        return None
    
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    embedding_model = "BAAI/bge-large-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

## Generation function
def llm_generate(llm, prompt):
  template = ChatPromptTemplate.from_messages([
      ("system", "Eres un experto en marketing digital especializado en SEO y redacción persuasiva."),
      ("human", "{prompt}"),
  ])

  chain = template | llm | StrOutputParser()

  res = chain.invoke({"prompt": prompt})
  return res

st.sidebar.title("Configuración")
api_key = st.sidebar.text_input("Introduce tu API Key de Groq:", type="password")
app_mode = st.sidebar.selectbox("Selecciona el modo:", ["Generador de Contenido", "Atención al Cliente"])

if app_mode == "Generador de Contenido":
    st.title("Generador de Contenido")

    topic = st.text_input("Tema:", placeholder="ej., nutrición, salud mental, chequeos de rutina, consejos de autocuidado, etc.")
    platform = st.selectbox("Plataforma:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'Correo Electrónico'])
    tone = st.selectbox("Tono del mensaje:", ['Normal', 'Informativo', 'Inspirador', 'Urgente', 'Informal'])
    length = st.selectbox("Longitud del texto:", ['Corto', 'Medio', 'Largo'])
    audience = st.selectbox("Público objetivo:", ['Todos', 'Jóvenes adultos', 'Familias', 'Personas mayores', 'Adolescentes'])
    cta = st.checkbox("Incluir llamada a la acción (CTA)")
    hashtags = st.checkbox("Incluir Hashtags")
    keywords = st.text_area("Palabras clave (SEO):", placeholder="Ejemplo: bienestar, salud preventiva...")

    if st.button("Generar Contenido"):
        if not api_key:
            st.error("Por favor, introduce tu API Key de Groq en el menú lateral.")
        else:
            prompt = f"""
            Escribe un texto optimizado para SEO sobre el tema '{topic}'.
            Devuelve solo el texto final en tu respuesta y no lo pongas entre comillas.
            - Plataforma de publicación: {platform}.
            - Tono: {tone}.
            - Público objetivo: {audience}.
            - Longitud: {length}.
            - {"Incluye una llamada a la acción (CTA) clara." if cta else "No incluyas una llamada a la acción."}
            - {"Incluye hashtags relevantes al final del texto." if hashtags else "No incluyas hashtags."}
            {"- Palabras clave a incluir (para SEO): " + keywords if keywords else ""}
            Responde siempre en español.
            """
            try:
                llm = get_llm(api_key)
                res = llm_generate(llm, prompt)
                st.markdown(res)
            except Exception as e:
                st.error(f"Error: {e}")

elif app_mode == "Atención al Cliente":
    st.title("Atención al Cliente 🏦")
    if not api_key:
        st.error("Por favor, introduce tu API Key de Groq en el menú lateral.")
    else:
        # Initialize RAG
        with st.spinner("Cargando base de conocimientos..."):
            vectorstore = initialize_rag()
        
        if vectorstore is None:
             st.error("No se encontró el archivo 'safebank-manual.pdf'. Por favor, súbelo a la carpeta del proyecto.")
        else:
            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("¿En qué puedo ayudarte?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    # RAG Logic
                    try:
                        llm = get_llm(api_key)
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                        
                        template_rag = """
                        You are a helpful virtual assistant answering general questions about a company's services.
                        Use the following bits of retrieved context to answer the question.
                        If you don't know the answer, just say you don't know. Keep your answer concise.
                        
                        Question: {input}
                        Context: {context}
                        
                        Answer:
                        """
                        prompt_rag = PromptTemplate.from_template(template_rag)
                        
                        chain_rag = (
                            {"context": retriever, "input": RunnablePassthrough()}
                            | prompt_rag
                            | llm
                            | StrOutputParser()
                        )
                        
                        response = chain_rag.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error al generar respuesta: {e}")
