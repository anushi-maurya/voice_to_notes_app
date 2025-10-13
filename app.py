import streamlit as st
import tempfile, os, textwrap, re
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from faster_whisper import WhisperModel
from transformers import pipeline
import torch

# -------------------------------
# Load models (cached in Streamlit)
# -------------------------------
@st.cache_resource
def load_models():
    device = "cpu"  # Streamlit Cloud runs on CPU only

    # Use smaller models for faster load
    whisper_model = WhisperModel("tiny", device=device)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    flash_gen = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

    return whisper_model, summarizer, flash_gen, device


# -------------------------------
# Transcription Function
# -------------------------------
def transcribe_with_faster_whisper(model, audio_path):
    # Handle both new & old versions of faster-whisper
    result = model.transcribe(audio_path, beam_size=5)

    if isinstance(result, tuple) and len(result) == 3:
        segments, info, text = result
    elif isinstance(result, tuple) and len(result) == 2:
        segments, info = result
        text = " ".join([s.text.strip() for s in segments if s.text])
    else:
        raise ValueError("Unexpected return format from faster-whisper model")

    transcript = text.strip()
    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    return segs, transcript


# -------------------------------
# Text Chunking for Summarization
# -------------------------------
def chunk_text(text, max_chars=2000):
    sentences = text.split(". ")
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 2 <= max_chars:
            cur = cur + (". " if cur else "") + s
        else:
            chunks.append(cur.strip() + ".")
            cur = s
    if cur:
        chunks.append(cur.strip() + ".")
    return chunks


# -------------------------------
# Hierarchical Summarization
# -------------------------------
def hierarchical_summarize(summarizer, text, chunk_size=1800):
    chunks = chunk_text(text, max_chars=chunk_size)
    chunk_summaries = []
    for c in chunks:
        try:
            s = summarizer(c, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        except Exception:
            s = c[:500]
        chunk_summaries.append(s)

    joined = " ".join(chunk_summaries)
    if len(joined) < 50:
        final = joined
    else:
        final = summarizer(joined, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    return final, chunk_summaries


# -------------------------------
# Flashcard Generator
# -------------------------------
def generate_flashcards(flash_gen, text, n=8):
    prompt = (
        f"Generate {n} concise flashcards (format: Q: question? A: short answer.) "
        "from the following lecture transcript. Keep questions focused and answers short.\n\nTranscript:\n"
        + text
    )
    res = flash_gen(prompt, max_length=512, do_sample=False)[0]["generated_text"]

    cards = []
    for line in res.replace("\r", "\n").split("\n"):
        line = line.strip()
        if line.startswith("Q:") or line.startswith("q:"):
            if "A:" in line:
                parts = line.split("A:")
                q = parts[0].strip()
                a = "A:" + parts[1].strip() if len(parts) > 1 else ""
                cards.append((q, a))
    if not cards:
        parts = res.split("Q:")
        for p in parts[1:]:
            if "A:" in p:
                q, a = p.split("A:", 1)
                cards.append(("Q: " + q.strip(), "A: " + a.strip()))
    return cards[:n]


# -------------------------------
# PDF Generator
# -------------------------------
def clean_text_for_pdf(text, width=100):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return "\n".join(textwrap.wrap(text, width))


def save_pdf(filename, summary, transcript, flashcards, pdf_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Lecture Notes", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, clean_text_for_pdf(summary))

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Transcript", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    safe_transcript = transcript[:3000] + ("\n\n...truncated..." if len(transcript) > 3000 else "")
    pdf.multi_cell(0, 6, clean_text_for_pdf(safe_transcript))

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Flashcards", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for i, (q, a) in enumerate(flashcards):
        pdf.multi_cell(0, 6, f"Q{i+1}: {clean_text_for_pdf(q)}\nA: {clean_text_for_pdf(a)}\n")

    pdf.output(pdf_path)


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Voice-to-Notes", layout="wide")
st.title("🎙 Lecture Voice → Notes Converter")
st.write("Upload an audio file (mp3/wav/m4a). We'll transcribe, summarize, and generate flashcards.")

with st.expander("Model Loading Status"):
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if not st.session_state.models_loaded:
        st.info("Loading models (this happens once). Please wait...")
        try:
            whisper_model, summarizer, flash_gen, device = load_models()
            st.success(f"Models loaded — device: {device}")
            st.session_state.whisper_model = whisper_model
            st.session_state.summarizer = summarizer
            st.session_state.flash_gen = flash_gen
            st.session_state.device = device
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.stop()
    else:
        st.success(f"Models ready (device: {st.session_state.device})")

uploaded = st.file_uploader("Upload audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a", "ogg"])
if uploaded:
    with st.spinner("Processing audio..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/mp3")

    if st.button("Transcribe & Generate Notes"):
        with st.spinner("Transcribing..."):
            segments, transcript = transcribe_with_faster_whisper(st.session_state.whisper_model, tmp_path)

        st.subheader("Transcript")
        with st.expander("Full Transcript", expanded=False):
            st.write(transcript)

        st.subheader("Summary")
        with st.spinner("Summarizing..."):
            summary, chunk_summaries = hierarchical_summarize(st.session_state.summarizer, transcript)
        st.write(summary)

        st.subheader("Flashcards")
        with st.spinner("Generating flashcards..."):
            flashcards = generate_flashcards(st.session_state.flash_gen, summary if len(summary) > 100 else transcript, n=8)
        for q, a in flashcards:
            st.markdown(f"**{q}**\n\n{a}")

        # Download buttons
        notes_text = f"Title: {uploaded.name}\n\nSummary:\n{summary}\n\nFlashcards:\n"
        for q, a in flashcards:
            notes_text += f"{q}\n{a}\n\n"
        notes_text += "\n\nTranscript:\n" + transcript

        st.download_button("Download notes (.txt)", notes_text, file_name="lecture_notes.txt")

        pdf_path = os.path.join(tempfile.gettempdir(), "lecture_notes.pdf")
        save_pdf(uploaded.name, summary, transcript, flashcards, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download notes (.pdf)", f, file_name="lecture_notes.pdf")
