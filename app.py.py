import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import os
import time

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Mic To Text - Neurointegrando",
    page_icon="📋",
    layout="centered"
)

# --- CSS: CUSTOMIZAÇÃO DO BOTÃO DE GRAVAÇÃO ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* 1. FUNDO GERAL */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7f6 !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(30, 58, 90, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(108, 117, 125, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(30, 58, 90, 0.04) 0%, transparent 60%) !important;
        background-attachment: fixed;
    }

    /* 2. ESTICAR O BOTÃO DE GRAVAR E ARREDONDAR CANTOS */
    /* Aqui a gente "domina" a barra branca e a deixa bonita */
    div.element-container:has(iframe) {
        background-color: white !important;
        border-radius: 20px !important; /* Cantos bem arredondados */
        padding: 10px !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        display: flex;
        justify-content: center;
        width: 100% !important; /* Estica para ocupar a largura do card */
    }

    iframe {
        border-radius: 15px !important;
    }

    /* 3. ÁREA DE TEXTO E RESULTADOS */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #ced4da !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a2a3a !important;
        font-size: 16px !important;
    }

    div[data-testid="stVerticalBlock"] > div:has(div.stAlert) {
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* 4. BOTÃO LIMPAR */
    .stButton>button {
        width: 140px !important;
        border-radius: 12px !important;
        background-color: white !important;
        color: #1e3a5a !important;
        font-weight: 700 !important;
        border: 2px solid #1e3a5a !important;
    }
    
    .stButton>button:hover {
        background-color: #1e3a5a !important;
        color: white !important;
    }

    h1, h2, h3, p, label, span { color: #1e3a5a !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

# --- CONTEÚDO ---
st.markdown('<div class="main-header" style="text-align:center;"><h1>Mic To Text</h1><p>Neurointegrando</p></div>', unsafe_allow_html=True)

st.info("🎙️ **Transcrever Evoluções**: Faça o relato da sessão de hoje.")

st.markdown("### 1. Relato da Evolução (Fale agora)")

if 'recorder_key' not in st.session_state:
    st.session_state.recorder_key = 0

# Gravador ocupando a largura disponível
audio_data = mic_recorder(
    start_prompt="🔴 Iniciar Gravação de Voz",
    stop_prompt="⏹️ Finalizar e Processar",
    key=f'recorder_{st.session_state.recorder_key}',
    just_once=True
)

if audio_data:
    st.audio(audio_data['bytes'])
    
    with st.spinner("🤖 Processando fala..."):
        temp_file = f"temp_{int(time.time())}.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data['bytes'])
        
        contexto_clinico = "Evolução clínica, prontuário, paciente, terapia, sessão, conduta terapêutica, TEA, ABA, psicóloga, Neurointegrando, desenvolvimento infantil."
        
        result = model.transcribe(
            temp_file, 
            fp16=False, 
            language="pt",
            temperature=0.0,
            initial_prompt=contexto_clinico
        )
        
        texto_gerado = result["text"].strip()
        
        if os.path.exists(temp_file):
            os.remove(temp_file)

    st.markdown("### 2. Resultado da Transcrição")
    texto_final = st.text_area(
        label="Texto pronto para copiar:",
        value=texto_gerado,
        height=350
    )

    if st.button("🗑️ Limpar"):
        st.session_state.recorder_key += 1
        st.rerun()

    st.success("💡 Dica: Selecione o texto, copie e cole na evolução do paciente (Átrio).")
else:
    st.write("---")
    st.caption("Aguardando gravação...")

# --- FOOTER ---
st.markdown("<br><hr><div style='text-align: center; color: #6c757d; font-family: Inter, sans-serif;'><p>🔒 Sua segurança é nossa prioridade. Nenhum arquivo é armazenado.</p><p><i>Feito com Carinho pela Equipe Administrativa da Clinica Neurointegrando.</i></p></div>", unsafe_allow_html=True)
