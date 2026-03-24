import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import os
import time

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Neuro | Mic To Text",
    page_icon="🎙️",
    layout="centered"
)

# --- CSS: ESTÉTICA APPLE (MINIMALISTA PREMIUM) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* 1. FUNDO E TIPOGRAFIA BASE */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #ffffff !important;
        color: #1d1d1f !important; /* Cor de texto padrão Apple */
    }

    /* 2. TÍTULOS IMPACTANTES */
    .main-title {
        font-weight: 700;
        font-size: 52px;
        letter-spacing: -1.5px;
        text-align: center;
        margin-top: 40px;
        color: #1d1d1f;
        margin-bottom: 5px;
    }
    .sub-title {
        font-weight: 400;
        font-size: 22px;
        color: #86868b;
        text-align: center;
        margin-bottom: 50px;
        letter-spacing: -0.5px;
    }

    /* 3. CAMUFLAGEM DO GRAVADOR E CARDS */
    div.element-container:has(iframe) {
        background-color: transparent !important;
        border: none !important;
        padding: 10px 0 !important;
    }

    /* Estilização da Área de Texto (Soft Shadow) */
    .stTextArea textarea {
        border-radius: 18px !important;
        border: 1px solid #d2d2d7 !important;
        background-color: #fbfbfd !important;
        color: #1d1d1f !important;
        font-size: 17px !important;
        padding: 20px !important;
        line-height: 1.5 !important;
        box-shadow: none !important;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #0071e3 !important;
    }

    /* 4. BOTÕES ESTILO macOS/iOS */
    .stButton>button {
        width: 140px !important;
        border-radius: 12px !important;
        height: 44px !important;
        background-color: #0071e3 !important; /* Azul Clássico Apple */
        color: white !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0077ed !important;
        opacity: 0.9;
    }

    /* Botão secundário (Lixeira/Limpar) */
    .stButton:has(button:contains("Limpar")) button {
        background-color: #f5f5f7 !important;
        color: #0071e3 !important;
        border: 1px solid #d2d2d7 !important;
    }

    /* Customização do componente st.info e st.success para serem mais discretos */
    .stAlert {
        background-color: #f5f5f7 !important;
        border: none !important;
        border-radius: 14px !important;
        color: #1d1d1f !important;
    }

    /* Esconder bordas desnecessárias */
    hr {
        border-top: 1px solid #d2d2d7 !important;
        opacity: 0.5;
    }

    h3 {
        font-weight: 600 !important;
        color: #1d1d1f !important;
        letter-spacing: -0.5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

# --- CONTEÚDO VISUAL ---
st.markdown('<h1 class="main-title">Mic To Text.</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sua voz transformada em evolução clínica.</p>', unsafe_allow_html=True)

# Container Centralizado de Instrução
st.info("🎙️ **Evoluções em tempo real.** O sistema utiliza IA para transcrever termos técnicos com precisão.")

st.markdown("### 1. Gravar Relato")

if 'recorder_key' not in st.session_state:
    st.session_state.recorder_key = 0

# Gravador Camuflado no fundo branco
audio_data = mic_recorder(
    start_prompt="🔴 Iniciar Gravação",
    stop_prompt="⏹️ Finalizar e Processar",
    key=f"recorder_{st.session_state.recorder_key}",
    just_once=True
)

if audio_data:
    st.audio(audio_data['bytes'])
    
    with st.spinner("Analisando áudio..."):
        temp_file = f"temp_{int(time.time())}.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data['bytes'])
        
        contexto = "Evolução clínica, prontuário, paciente, terapia, sessão, conduta, TEA, ABA, psicóloga, Neurointegrando, desenvolvimento infantil."
        result = model.transcribe(temp_file, fp16=False, language="pt", temperature=0.0, initial_prompt=contexto)
        
        texto_gerado = result["text"].strip()
        if os.path.exists(temp_file):
            os.remove(temp_file)

    st.markdown("### 2. Texto Gerado")
    st.text_area(label="", value=texto_gerado, height=350, label_visibility="collapsed")

    # Layout de botões
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Limpar"):
            st.session_state.recorder_key += 1
            st.rerun()
    
    st.success("💡 **Dica:** Copie o texto e cole diretamente no sistema (Átrio).")
else:
    st.write("---")
    st.caption("Aguardando entrada de áudio para processamento.")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #86868b; font-size: 13px;">
        <p>Copyright © 2026 Clínica Neurointegrando. Todos os direitos reservados.</p>
        <p>🔒 Segurança de nível bancário: seus dados são processados localmente e não são armazenados.</p>
    </div>
""", unsafe_allow_html=True)
