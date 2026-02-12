import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
