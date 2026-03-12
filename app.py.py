import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import os
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Mic To Text - Neurointegrando",
    page_icon="📋",
    layout="centered"
)

# --- CSS: FORÇANDO TEMA CLARO, CORES E TEXTO ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* 1. FORÇAR FUNDO BRANCO E CORES GERAIS */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7f6 !important;
        color: #1a2a3a !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(30, 58, 90, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(108, 117, 125, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(30, 58, 90, 0.04) 0%, transparent 60%) !important;
        background-attachment: fixed;
    }

    /* 2. AJUSTE DO TEXTO NA ÁREA DE TRANSCRIÇÃO (RESOLVE O TEXTO BRANCO) */
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #ced4da !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a2a3a !important; /* Cor azul escuro/preto */
        font-size: 16px !important;
        -webkit-text-fill-color: #1a2a3a !important; /* Força cor no iOS/Safari */
    }

    /* Ajuste do placeholder (texto que aparece antes de digitar) */
    .stTextArea textarea::placeholder {
        color: #6c757d !important;
    }

    /* 3. EFEITO VIDRO NOS CONTAINERS */
    div[data-testid="stVerticalBlock"] > div:has(div.stAlert) {
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Cabeçalho e Títulos */
    .main-header {
        text-align: center;
        padding: 20px 0;
        color: #1e3a5a !important;
    }
    
    h1, h2, h3, p, label {
        color: #1e3a5a !important;
    }

    /* Botões */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        background-color: #1e3a5a;
        color: white !important;
        font-weight: 600;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #2c5282;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

# --- CONTEÚDO ---
st.markdown('<div class="main-header"><h1>Mic To Text</h1><p>Neurointegrando</p></div>', unsafe_allow_html=True)

st.info("🎙️ **Transcrever Evoluções**: Faça o relato da sessão de hoje.")

st.markdown("### 1. Relato da Evolução (Fale agora)")
audio_data = mic_recorder(
    start_prompt="🔴 Iniciar Gravação de Voz",
    stop_prompt="⏹️ Finalizar e Processar",
    key='recorder',
    just_once=True
)

if audio_data:
    st.audio(audio_data['bytes'])
    
    with st.spinner("🤖 Processando fala com alta precisão..."):
        temp_file = "temp_audio_clinica.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data['bytes'])
        
        contexto_clinico = "Evolução clínica, prontuário, paciente, terapia, sessão, conduta terapêutica, queixa, desenvolvimento, TEA, ABA, psicóloga, Neurointegrando."
        
        result = model.transcribe(
            temp_file, 
            fp16=False, 
            language="pt",
            temperature=0.0,
            initial_prompt=contexto_clinico
        )
        
        texto_gerado = result["text"].strip()
        os.remove(temp_file)

    st.markdown("### 2. Resultado da Transcrição")
    texto_final = st.text_area(
        label="Texto pronto para copiar:",
        value=texto_gerado,
        height=350
    )

    if st.button("🗑️ Limpar"):
        st.rerun()

    st.success("💡 Dica: Selecione o texto, copie e cole na evolução do paciente (Átrio).")
else:
    st.write("---")
    st.caption("Aguardando gravação para gerar o relatório...")

# --- FOOTER ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #6c757d; font-family: 'Inter', sans-serif;">
        <p style="margin-bottom: 5px;">
            🔒 Sua segurança é nossa prioridade. Nenhum arquivo é enviado ou armazenado.
        </p>
        <p style="font-style: italic; font-size: 0.9em;">
            Feito com Carinho pela Equipe Administrativa da Clinica Neurointegrando.
        </p>
    </div>
    """, unsafe_allow_html=True)
