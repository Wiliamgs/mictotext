import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
import os
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Transcrição de Evoluções | Neurointegrando",
    page_icon="📋",
    layout="wide"
)

# --- ESTILO CSS (Inspirado no seu Gerador de Relatórios) ---
st.markdown("""
    <style>
    /* Fundo e Container */
    .main { background-color: #f8f9fa; }
    
    /* Estilização dos Blocos */
    .stAlert { border-radius: 10px; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    
    /* Botões */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        background-color: #2E4053;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B2631;
        border: 1px solid #2E4053;
    }
    
    /* Área de Texto */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #D5DBDB !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- HEADER ---
st.title("📋 Assistente de Evolução Clínica")
st.markdown(f"**Data:** {datetime.now().strftime('%d/%m/%Y')} | **Usuário:** Terapeuta")
st.divider()

# --- LAYOUT EM COLUNAS ---
col_gravacao, col_resultado = st.columns([1, 2], gap="large")

with col_gravacao:
    with st.container():
        st.subheader("🎤 Captura de Áudio")
        st.info("Clique para iniciar o relato da sessão. Tente falar de forma clara.")
        
        # O gravador
        audio_data = mic_recorder(
            start_prompt="🔴 Iniciar Relato",
            stop_prompt="⏹️ Finalizar Gravação",
            key='recorder',
            just_once=True
        )
        
        if audio_data:
            st.success("✅ Áudio capturado com sucesso!")
            st.audio(audio_data['bytes'])

with col_resultado:
    st.subheader("📝 Evolução Gerada")
    
    if audio_data:
        with st.spinner("🤖 A IA está transcrevendo sua fala..."):
            # Processamento
            temp_file = "temp_evolucao.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_data['bytes'])
            
            result = model.transcribe(temp_file, fp16=False, language="pt")
            texto_transcrito = result["text"].strip()
            os.remove(temp_file)
        
        # Campo de texto para edição
        texto_final = st.text_area(
            "Revise e ajuste o texto abaixo:",
            value=texto_transcrito,
            height=350
        )
        
        # Ações do Relatório
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Limpar e Reiniciar"):
                st.rerun()
        with c2:
            st.markdown("💡 **Dica:** Use `Ctrl+C` para copiar e colar no sistema da clínica.")
    else:
        # Placeholder quando não há áudio
        st.write("---")
        st.info("Aguardando gravação para processar a evolução...")

# --- FOOTER ---
st.divider()
st.caption("Sistema Interno de Apoio ao Terapeuta - Clínica Neurointegrando")