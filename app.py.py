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

# --- CSS: FUNDO COM BOLAS, MODO CLARO FORÇADO E CORREÇÃO DE BUGS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* 1. FORÇAR FUNDO CLARO E BOLAS TRANSPARENTES */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #f4f7f6 !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(30, 58, 90, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(108, 117, 125, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(30, 58, 90, 0.04) 0%, transparent 60%) !important;
        background-attachment: fixed;
        color: #1a2a3a !important;
    }

    /* 2. REMOVER BARRA BRANCA DO BOTÃO DE GRAVAR */
    div[data-testid="stVerticalBlock"] > div:has(button[id*="recorder"]) {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    /* Garantir transparência no componente de áudio */
    iframe {
        background-color: transparent !important;
    }

    /* 3. ÁREA DE TEXTO (TEXTO ESCURO E FUNDO CLARO) */
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #ced4da !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a2a3a !important;
        font-size: 16px !important;
        -webkit-text-fill-color: #1a2a3a !important;
    }

    /* 4. EFEITO VIDRO NOS CONTAINERS (DICA E RESULTADO) */
    div[data-testid="stVerticalBlock"] > div:has(div.stAlert) {
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* 5. BOTÃO DE REINICIAR (LIXEIRA) - CORREÇÃO DE COR */
    .stButton>button {
        width: 100px; /* Tamanho menor para a lixeira não ocupar a tela toda */
        border-radius: 8px;
        height: 45px;
        background-color: #1e3a5a !important;
        color: white !important;
        font-weight: 600;
        border: 2px solid #1e3a5a !important;
        transition: 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stButton>button:hover {
        background-color: #2c5282 !important;
        border-color: #2c5282 !important;
        color: white !important;
    }

    /* Títulos e Textos */
    .main-header {
        text-align: center;
        padding: 20px 0;
        color: #1e3a5a !important;
    }
    h1, h2, h3, p, label, span {
        color: #1e3a5a !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO (SMALL) ---
@st.cache_resource
def load_model():
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
        
        # Contexto para melhorar a precisão dos termos da clínica
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

# --- FOOTER (IDÊNTICO À IMAGEM SOLICITADA) ---
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
