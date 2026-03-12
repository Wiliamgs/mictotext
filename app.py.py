import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import os
import time

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Mic To Text - Neurointegrando",
    page_icon="🎙️",
    layout="centered"
)

# --- CSS: CAMUFLAGEM E ESTILO ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* 1. FUNDO BRANCO PARA CAMUFLAR O COMPONENTE DE ÁUDIO */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff !important;
        color: #1a2a3a !important;
    }

    /* 2. REINSERINDO AS BOLAS DA CLÍNICA COMO DETALHE SUAVE NO FUNDO */
    [data-testid="stAppViewContainer"] {
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(30, 58, 90, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(108, 117, 125, 0.05) 0%, transparent 40%) !important;
        background-attachment: fixed;
    }

    /* 3. LIMPANDO O COMPONENTE DE GRAVAÇÃO (MATA A BARRA BRANCA) */
    div.element-container:has(iframe) {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    iframe {
        border: none !important;
        background-color: transparent !important;
    }

    /* 4. ÁREA DE TEXTO */
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        background-color: #ffffff !important;
        color: #1a2a3a !important;
        font-size: 16px !important;
    }

    /* 5. BOTÃO LIMPAR (AZUL E BRANCO) */
    .stButton>button {
        width: 140px !important;
        border-radius: 8px !important;
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
st.markdown('<div style="text-align:center;"><h1>Mic To Text</h1><p>Neurointegrando</p></div>', unsafe_allow_html=True)

st.info("🎙️ **Transcrever Evoluções**: Faça o relato da sessão de hoje.")

st.markdown("### 1. Relato da Evolução (Fale agora)")

if 'recorder_key' not in st.session_state:
    st.session_state.recorder_key = 0

# Gravador
audio_data = mic_recorder(
    start_prompt="🔴 Iniciar Gravação de Voz",
    stop_prompt="⏹️ Finalizar e Processar",
    key=f"recorder_{st.session_state.recorder_key}",
    just_once=True
)

if audio_data:
    st.audio(audio_data['bytes'])
    
    with st.spinner("🤖 Processando fala..."):
        temp_file = f"temp_{int(time.time())}.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data['bytes'])
        
        contexto = "Evolução clínica, prontuário, paciente, terapia, sessão, conduta, TEA, ABA, psicóloga, Neurointegrando."
        result = model.transcribe(temp_file, fp16=False, language="pt", temperature=0.0, initial_prompt=contexto)
        
        texto_gerado = result["text"].strip()
        if os.path.exists(temp_file):
            os.remove(temp_file)

    st.markdown("### 2. Resultado da Transcrição")
    st.text_area(label="Texto pronto para copiar:", value=texto_gerado, height=350)

    if st.button("🗑️ Limpar"):
        st.session_state.recorder_key += 1
        st.rerun()
    
    st.success("💡 Dica: Selecione o texto, copie e cole na evolução (Átrio).")
else:
    st.write("---")
    st.caption("Aguardando gravação...")

# --- FOOTER (CORRIGIDO) ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #6c757d;">
        <p>🔒 Sua segurança é nossa prioridade. Nenhum arquivo é armazenado.</p>
        <p><i>Feito com Carinho pela Equipe Administrativa da Clinica Neurointegrando.</i></p>
    </div>
""", unsafe_allow_html=True)
