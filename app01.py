Conchita_EDEM_LLM_AI_App_for_Business_Marketing.ipynb
ipynb

safebank-manual.pdf
pdf


Conchita_EDEM_LLMs_Customer_Service_and_Support.ipynb
ipynb

Con estos archivos y con esos codigos debo generar una pagina en streamlit en el que tenga un archivo .py y el requirement.txt que suba a gitbug para que se pueda ver. La idea es generar con al estructura del documento esa web en la que uno introduzca una clave API de Grog, cumplimente todos los campos y el modelo responda segun el tema y los parametros elegidos contenido para esas plataformas. Despues por otro lado, con el documento pdf adjunto se debe poder tener una opcion para atención al cliente donde se respondan preguntar sobre esa pdf

12:30
Voy a analizar los archivos y crear una aplicación Streamlit completa con las dos funcionalidades solicitadas.


Ejecutó 3 comandos, visualizó un archivo
Analizo los archivos para crear la aplicación Streamlit completa. Voy a generar una app con dos funcionalidades: generador de contenido de marketing y chatbot de atención al cliente basado en el PDF.


Se crearon 5 archivos, ejecutó un comando
Ahora voy a mover todos los archivos al directorio de outputs para que puedas acceder a ellos:


Ejecutó un comando, leyó un archivo
ANÁLISIS COMPLETADO
Confianza: 95%

Tipo de respuesta: Técnica/Factual con implementación práctica

Resumen ejecutivo: He creado una aplicación Streamlit completa con dos funcionalidades integradas:

📦 Archivos generados:
app.py - Aplicación principal con:
Tab 1: Generador de contenido de marketing (basado en notebook)
Tab 2: Chatbot de atención al cliente (basado en PDF SafeBank)
Sistema de tabs para navegación
Configuración de API Key en sidebar
UI mejorada con CSS personalizado
requirements.txt - Dependencias necesarias
README.md - Documentación completa de uso
DEPLOYMENT.md - Guía de despliegue en Streamlit Cloud
.gitignore - Protección de archivos sensibles
safebank-manual.pdf - Base de conocimiento para chatbot
🎯 Funcionalidades implementadas:
Marketing Generator:

Campos configurables (tema, plataforma, tono, longitud, audiencia)
Opciones de CTA y hashtags
Keywords SEO
Descarga del contenido generado
Customer Service Bot:

RAG con FAISS y embeddings multilingües
Memoria conversacional
Interfaz de chat interactiva
Base de conocimiento del PDF SafeBank
🚀 Próximos pasos:
Subir archivos a GitHub
Obtener API Key de Groq
Ejecutar localmente: streamlit run app.py
Desplegar en Streamlit Cloud (opcional)
Principal incertidumbre: Compatibilidad de versiones de librerías en diferentes entornos.

Información faltante: Tu repositorio de GitHub específico para incluir en las instrucciones.

App
PY 
Requirements
TXT 
Readme
Documento · MD 
Deployment
Documento · MD 
.gitignore
GITIGNORE 
Safebank manual
Documento · PDF 


¿Quieres recibir una notificación cuando Claude responda?



Claude es IA y puede cometer errores. Por favor, verifica las respuestas.
App · PY
Copiar

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuración de página
st.set_page_config(
    page_title="AI Business App - Marketing & Customer Service",
    page_icon="🤖",
    layout="wide"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🤖 AI Business Application</h1>', unsafe_allow_html=True)

# Sidebar para API Key
with st.sidebar:
    st.markdown("### 🔑 Configuración")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Ingresa tu API key de Groq",
        help="Obtén tu API key en https://console.groq.com"
    )
    
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("✅ API Key configurada")
    else:
        st.warning("⚠️ Por favor ingresa tu API Key")
    
    st.markdown("---")
    st.markdown("### 📚 Información")
    st.info("""
    **Marketing Content Generator**: Crea contenido optimizado para diferentes plataformas.
    
    **Customer Service**: Chatbot basado en el manual de SafeBank.
    """)

# Inicializar LLM
@st.cache_resource
def init_llm(api_key):
    if not api_key:
        return None
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

# Función de generación para marketing
def generate_marketing_content(llm, prompt):
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a digital marketing expert specialized in SEO and persuasive copywriting."),
        ("human", "{prompt}"),
    ])
    
    chain = template | llm | StrOutputParser()
    res = chain.invoke({"prompt": prompt})
    return res

# Función para cargar y procesar PDF
@st.cache_resource
def load_pdf_knowledge(_llm):
    # Cargar PDF
    loader = PyPDFLoader("safebank-manual.pdf")
    documents = loader.load()
    
    # Dividir en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Crear embeddings y vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Crear cadena conversacional
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain

# Tabs principales
tab1, tab2 = st.tabs(["📢 Marketing Content Generator", "💬 Customer Service Bot"])

# TAB 1: Marketing Content Generator
with tab1:
    st.markdown('<h2 class="sub-header">Generador de Contenido para Marketing</h2>', unsafe_allow_html=True)
    
    if not api_key:
        st.warning("⚠️ Por favor configura tu API Key en el panel lateral para usar esta funcionalidad.")
    else:
        llm = init_llm(api_key)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            topic = st.text_input(
                "📝 Tema del contenido:",
                placeholder="ej: nutrición, salud mental, tecnología, fitness...",
                help="El tema principal sobre el que quieres crear contenido"
            )
        
        with col2:
            platform = st.selectbox(
                "📱 Plataforma:",
                ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'Email']
            )
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            tone = st.selectbox(
                "🎭 Tono del mensaje:",
                ['Normal', 'Informativo', 'Inspirador', 'Urgente', 'Informal']
            )
        
        with col4:
            length = st.selectbox(
                "📏 Longitud:",
                ['Corto', 'Medio', 'Largo']
            )
        
        with col5:
            audience = st.selectbox(
                "👥 Audiencia:",
                ['Todos', 'Jóvenes adultos', 'Familias', 'Adultos mayores', 'Adolescentes']
            )
        
        col6, col7 = st.columns(2)
        
        with col6:
            cta = st.checkbox("✅ Incluir CTA (Call to Action)")
        
        with col7:
            hashtags = st.checkbox("🏷️ Incluir Hashtags")
        
        keywords = st.text_area(
            "🔍 Keywords (SEO):",
            placeholder="Ejemplo: bienestar, salud preventiva, tecnología innovadora...",
            help="Palabras clave para optimización SEO"
        )
        
        if st.button("🚀 Generar Contenido", type="primary", use_container_width=True):
            if not topic:
                st.error("❌ Por favor ingresa un tema para generar contenido.")
            else:
                with st.spinner("⏳ Generando contenido..."):
                    prompt = f"""
                    Escribe un texto optimizado para SEO sobre el tema '{topic}'.
                    Devuelve solo el texto final en tu respuesta sin comillas.
                    - Plataforma donde se publicará: {platform}.
                    - Tono: {tone}.
                    - Audiencia objetivo: {audience}.
                    - Longitud: {length}.
                    - {"Incluye un Call to Action claro." if cta else "No incluyas Call to Action."}
                    - {"Incluye hashtags relevantes al final del texto." if hashtags else "No incluyas hashtags."}
                    {"- Keywords a incluir (para SEO): " + keywords if keywords else ""}
                    """
                    try:
                        result = generate_marketing_content(llm, prompt)
                        st.markdown("### 📄 Contenido Generado:")
                        st.markdown(f"```\n{result}\n```")
                        st.success("✅ ¡Contenido generado exitosamente!")
                        
                        # Botón de copiar
                        st.download_button(
                            label="📥 Descargar como TXT",
                            data=result,
                            file_name=f"contenido_{platform.lower()}_{topic.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# TAB 2: Customer Service Bot
with tab2:
    st.markdown('<h2 class="sub-header">Chatbot de Atención al Cliente - SafeBank</h2>', unsafe_allow_html=True)
    
    if not api_key:
        st.warning("⚠️ Por favor configura tu API Key en el panel lateral para usar esta funcionalidad.")
    else:
        # Verificar si existe el PDF
        pdf_path = "safebank-manual.pdf"
        if not os.path.exists(pdf_path):
            st.error(f"❌ No se encontró el archivo '{pdf_path}'. Por favor asegúrate de que esté en el mismo directorio que app.py")
        else:
            llm = init_llm(api_key)
            
            # Inicializar chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Cargar conocimiento del PDF
            try:
                with st.spinner("📚 Cargando base de conocimiento de SafeBank..."):
                    qa_chain = load_pdf_knowledge(llm)
                
                st.success("✅ Base de conocimiento cargada correctamente")
                
                # Mostrar historial de chat
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Input del usuario
                if prompt := st.chat_input("Escribe tu pregunta sobre SafeBank..."):
                    # Agregar mensaje del usuario
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generar respuesta
                    with st.chat_message("assistant"):
                        with st.spinner("🤔 Pensando..."):
                            response = qa_chain({"question": prompt})
                            answer = response["answer"]
                            st.markdown(answer)
                    
                    # Agregar respuesta al historial
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Botón para limpiar chat
                if st.button("🗑️ Limpiar conversación"):
                    st.session_state.messages = []
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error al cargar el PDF: {str(e)}")
                st.info("💡 Asegúrate de tener instaladas todas las dependencias necesarias.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Desarrollado con ❤️ usando Streamlit y Groq | © 2026</p>
</div>
""", unsafe_allow_html=True)
