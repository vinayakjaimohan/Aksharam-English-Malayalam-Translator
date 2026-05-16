import streamlit as st
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
MODEL_PATH = "./aksharam_model_final"

st.set_page_config(page_title="AKSHARAM", page_icon="🌐", layout="centered")

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = MBart50TokenizerFast.from_pretrained(BASE_MODEL)
    tokenizer.src_lang = "en_XX"

    base_model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return tokenizer, model, device


# -----------------------------
# TRANSLATE FUNCTION
# -----------------------------
def translate(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["ml_IN"],
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -----------------------------
# ORIGINAL UI STARTS HERE (UNCHANGED)
# -----------------------------
st.title("AKSHARAM")

english_text = st.text_area(
    label="English input",
    value="",
    height=180,
    placeholder="Type or paste your English text here…",
)

output_text = "Translation will appear here after generation."

# 🔥 LOAD MODEL ONCE
tokenizer, model, device = load_model()

if st.button("Translate"):
    if not english_text.strip():
        st.warning("Please enter some English text first.")
    else:
        with st.spinner("Translating…"):
            translation = translate(english_text, tokenizer, model, device)
            output_text = translation

        st.success("Output generated.")

st.text_area(
    label="Malayalam output",
    value=output_text,
    height=180,
)

st.stop()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Malayalam&family=Space+Mono:wght@400;700&display=swap');

/* Background */
.stApp { background: #0d0d0f; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Title block */
.title-block {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.title-block h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #f0ede6;
    letter-spacing: -0.02em;
    margin: 0;
}
.title-block p {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #5a5a6e;
    margin-top: 0.4rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Card wrapper */
.card {
    background: #16161a;
    border: 1px solid #2a2a35;
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
}

/* Label above textareas */
.area-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #5a5a6e;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Override Streamlit textarea */
textarea {
    background: #0d0d0f !important;
    border: 1px solid #2a2a35 !important;
    border-radius: 8px !important;
    color: #f0ede6 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
textarea:focus { border-color: #4f8ef7 !important; box-shadow: none !important; }
textarea::placeholder { color: #3a3a4a !important; }

/* Malayalam output */
.ml-output {
    font-family: 'Noto Serif Malayalam', serif;
    font-size: 1.3rem;
    color: #f0ede6;
    line-height: 1.8;
    padding: 1rem 0 0.2rem;
    min-height: 2.5rem;
}
.ml-placeholder {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #2a2a35;
    padding: 1rem 0 0.2rem;
}

/* Arrow divider */
.arrow-row {
    text-align: center;
    font-size: 1.4rem;
    color: #2a2a35;
    margin: -0.4rem 0;
    line-height: 1;
}

/* Translate button */
div.stButton > button {
    width: 100%;
    background: #4f8ef7;
    color: #0d0d0f;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    margin-top: 0.8rem;
    cursor: pointer;
    transition: background 0.2s;
}
div.stButton > button:hover { background: #3a7af0; }

/* Status badges */
.badge-gpu {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    background: #1a2a1a;
    color: #4ec94e;
    border: 1px solid #2a3a2a;
}
.badge-cpu {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    background: #2a1a10;
    color: #f09a4e;
    border: 1px solid #3a2a1a;
}

/* Spinner text */
.stSpinner > div { color: #4f8ef7 !important; }
</style>
""", unsafe_allow_html=True)