import streamlit as st
import ollama
import time

st.set_page_config(page_title="Estigia CubeSat Dashboard", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# Import the exact same model and logic from the main script!
from ollama_launch_2_1 import TelemetrySystem, PROMPTS, model

@st.cache_resource(show_spinner=True)
def get_telemetry():
    return TelemetrySystem()

telemetry = get_telemetry()

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.stChatMessage { border-radius: 10px; padding: 15px; margin-bottom: 10px; }
.stChatMessageAvatar { border-radius: 50%; }
.stMarkdown p { font-size: 1.05rem; }
.stExpanderHeader { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://www.upv.es/estudios/grado/images/logos/upv-logo.png", width=70)
with col2:
    st.title("🛰️ Estigia CubeSat - Pluton UPV")
    st.caption("Panel de Control y Chat de Telemetría (Ollama Powered)")

st.divider()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Configuración / Settings")
    st.info("Selecciona el idioma de interacción con Estigia.")
    
    lang_map = {"Español": "1", "Valencià": "2", "English": "3"}
    # Guardamos en session_state para evitar recalculos en vano
    lang_selection = st.radio("Idioma / Language", ["Español", "Valencià", "English"], index=0)
    lang_code = lang_map[lang_selection]

    st.divider()
    st.markdown("### 📊 Status Módulos")
    st.success("✅ Clasificador: " + "Cargado" if telemetry.classifier else "⚠️ Modo Fallback (Keywords)")
    st.success(f"🤖 LLM Ollama: {model}")

# Reset chat history if language changes
if "lang" not in st.session_state or st.session_state.lang != lang_code:
    st.session_state.lang = lang_code
    # Clear conversation history, but keep the initial System Prompt.
    st.session_state.messages = [{"role": "system", "content": PROMPTS[lang_code]["sys"]}]
    st.toast(f"Idioma cambiado a: {lang_selection}", icon="🔄")

# Display chat history (skip system prompt to avoid clutter)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"], avatar="🛰️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

# ---------- CHAT INPUT ----------
if prompt := st.chat_input("Escribe tu mensaje o comando de telemetría (ej: 'dame la temperatura')"):
    
    st.chat_message("user", avatar="👤").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 1. Prediction for Telemetry
    category = telemetry.predict(prompt)
    sensor_data = telemetry.get_data(category, lang_code)
    
    if sensor_data:
        # It's a telemetry request!
        with st.chat_message("assistant", avatar="📡"):
            st.info(f"**DATOS DE SENSOR RECIBIDOS:**\n\n{sensor_data}")
        st.session_state.messages.append({'role': 'assistant', 'content': sensor_data})
    else:
        # It's a standard chat so use Ollama
        with st.chat_message("assistant", avatar="🛰️"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # We need to build the history without duplicating the system prompt
                # Limit history to the last N messages + the single MUST HAVE system prompt
                current_chat = st.session_state.messages[1:] # everything except system prompt
                max_history = 4
                history_for_llm = [st.session_state.messages[0]] + current_chat[-(max_history*2):]
                
                response_stream = ollama.chat(
                    model=model,
                    messages=history_for_llm,
                    stream=True
                )
                for chunk in response_stream:
                    token = chunk['message']['content']
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Error conectando a Ollama: {e}")
                full_response = "Lo siento, tengo problemas de conexión con mis sistemas centrales."
                
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})

