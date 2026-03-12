import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import os
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA (PADRÃO NEUROINTEGRANDO) ---
st.set_page_config(
    page_title="Gerador de Relatórios - Neurointegrando",
    page_icon="📋",
    layout="centered"  # O site original é centralizado e não wide
)

# --- CSS PARA COPIAR O LAYOUT DO SITE ORIGINAL ---
st.markdown("""
    <style>
    /* Importando fonte similar à do site */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7f6; /* Fundo cinza claro do site original */
    }

    /* Cabeçalho principal */
    .main-header {
        text-align: center;
        padding: 20px 0;
        color: #1a2a3a;
    }

    /* Estilização dos Containers (Simulando os cards de upload) */
    div[data-testid="stVerticalBlock"] > div:has(div.stAlert) {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }

    /* Botão Principal (Azul escuro do site original) */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        background-color: #1e3a5a; /* Tom de azul do Gerador de Relatórios */
        color: white;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #2c5282;
        border: none;
        color: white;
    }

    /* Área de Texto (Evolução) */
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #ced4da !important;
        background-color: #ffffff !important;
    }

    /* Ajuste de labels e subtitulos */
    h1, h2, h3 {
        color: #1e3a5a !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO (WHISPER) ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- CONTEÚDO VISUAL (COPIANDO O HEADER DO SITE) ---
st.markdown('<div class="main-header"><h1>Gerador de Relatórios</h1><p>Neurointegrando</p></div>', unsafe_allow_html=True)

# Bloco de Instrução/Status
st.info("🎙️ **Transcritor de Evoluções**: Fale o relato da sessão abaixo para gerar o texto automaticamente.")

# --- ESTRUTURA DE PASSO A PASSO (Igual ao site original) ---
st.markdown("### 1. Relate a Evolução (Fale agora)")
audio_data = mic_recorder(
    start_prompt="🔴 Iniciar Gravação de Voz",
    stop_prompt="⏹️ Finalizar e Processar",
    key='recorder',
    just_once=True
)

if audio_data:
    st.audio(audio_data['bytes'])
    
    with st.spinner("Transformando sua fala em texto..."):
        temp_file = "temp_audio_clinica.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data['bytes'])
        
        # Transcrição
        result = model.transcribe(temp_file, fp16=False, language="pt")
        texto_gerado = result["text"].strip()
        os.remove(temp_file)

    st.markdown("### 2. Resultado da Transcrição")
    # Campo de texto para o terapeuta editar/copiar
    texto_final = st.text_area(
        label="Texto pronto para copiar e colar no prontuário:",
        value=texto_gerado,
        height=300
    )

    # Botão de ação (Inspirado no botão "Gerar e Baixar" do site original)
    if st.button("🗑️ Limpar para Novo Relato"):
        st.rerun()

    st.success("💡 Dica: Selecione o texto acima, use Ctrl+C e cole diretamente no sistema de evoluções.")
else:
    st.write("---")
    st.caption("Aguardando entrada de áudio para gerar o relatório...")

# --- FOOTER ---
st.markdown("<br><hr><center><p style='color: #6c757d;'>Sistema de Apoio Clínico - Neurointegrando</p></center>", unsafe_allow_html=True)
