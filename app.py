import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import os
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA (PADRÃO NEUROINTEGRANDO) ---
st.set_page_config(
    page_title="Mic To Text - Neurointegrando",
    page_icon="📋",
    layout="centered"
)

# --- CSS PARA COPIAR O LAYOUT DO SITE ORIGINAL ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7f6;
    }

    .main-header {
        text-align: center;
        padding: 20px 0;
        color: #1a2a3a;
    }

    div[data-testid="stVerticalBlock"] > div:has(div.stAlert) {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        background-color: #1e3a5a;
        color: white;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #2c5282;
        color: white;
    }

    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #ced4da !important;
        background-color: #ffffff !important;
        font-size: 16px;
    }

    h1, h2, h3 {
        color: #1e3a5a !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO (USANDO SMALL PARA MAIOR PRECISÃO) ---
@st.cache_resource
def load_model():
    # Trocamos para 'small' para resolver as 'coisas aleatórias' que as cobaias relataram
    return whisper.load_model("small")

model = load_model()

# --- CONTEÚDO VISUAL ---
st.markdown('<div class="main-header"><h1>Mic To Text</h1><p>Neurointegrando</p></div>', unsafe_allow_html=True)

st.info("🎙️ **Transcrever Evoluções**: Faça o relato da sessão de hoje.")

# --- PASSO 1: GRAVAÇÃO ---
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
        
        # PROMPT: Guia a IA para o vocabulário da clínica e evita erros comuns
        contexto_clinico = "Evolução clínica, prontuário, paciente, terapia, sessão, conduta terapêutica, queixa, desenvolvimento, TEA, ABA, psicóloga."
        
        # Transcrição Otimizada
        result = model.transcribe(
            temp_file, 
            fp16=False, 
            language="pt",
            temperature=0.0,
            initial_prompt=contexto_clinico
        )
        
        texto_gerado = result["text"].strip()
        os.remove(temp_file)

    # --- PASSO 2: RESULTADO ---
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
