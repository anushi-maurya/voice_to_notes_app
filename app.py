import streamlit as st
import tempfile, os, textwrap
import textwrap
import re
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos


# Load libraries inside function (so app starts even if imports take time)
@st.cache_resource
def load_models():
    # faster-whisper for transcription
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        st.error("faster-whisper not found or failed to import. Make sure it's installed.")
        raise

    from transformers import pipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Choose smaller models for demo and speed
    whisper_model = WhisperModel("small", device=device)  # try "tiny" or "small"
    # Summarizer: lightweight distilBART
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=0 if device == "cuda" else -1
    )
    # FLAN-T5 for instruction style text2text (flashcards)
    flash_gen = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=0 if device == "cuda" else -1
    )
    return whisper_model, summarizer, flash_gen, device

def transcribe_with_faster_whisper(model, audio_path):
    # returns segments list of dicts with start/end/text and a full transcript
    segments, info = model.transcribe(audio_path, beam_size=5)
    transcript = " ".join([s.text.strip() for s in segments if s.text])
    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    return segs, transcript

def chunk_text(text, max_chars=2000):
    # naive chunk by characters (keeps sentences intact roughly)
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

def hierarchical_summarize(summarizer, text, chunk_size=1800):
    chunks = chunk_text(text, max_chars=chunk_size)
    chunk_summaries = []
    for c in chunks:
        # summarizer can accept long text but we chunk for safety
        try:
            s = summarizer(c, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        except Exception:
            s = c[:500]  # fallback: truncate
        chunk_summaries.append(s)
    # summarize the summaries
    joined = " ".join(chunk_summaries)
    if len(joined) < 50:
        final = joined
    else:
        final = summarizer(joined, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    return final, chunk_summaries

def generate_flashcards(flash_gen, text, n=8):
    prompt = (
        "Generate " + str(n) + " concise flashcards (format: Q: question? A: short answer.) "
        "from the following lecture transcript. Keep questions focused and answers short.\n\nTranscript:\n"
        + text
    )
    res = flash_gen(prompt, max_length=512, do_sample=False)[0]["generated_text"]
    # parse lines that start with Q: or separate by A:
    cards = []
    # Normalize splitting: replace newlines with marker and split
    for line in res.replace("\r", "\n").split("\n"):
        line = line.strip()
        if line.startswith("Q:") or line.startswith("q:"):
            # attempt to find answer in same line after 'A:'
            if "A:" in line or "a:" in line:
                # split Q and A
                parts = line.split("A:")
                q = parts[0].strip()
                a = "A:" + parts[1].strip() if len(parts) > 1 else ""
                cards.append((q, a))
            else:
                # question only — try to attach next line as answer
                idx = res.splitlines().index(line)
                if idx + 1 < len(res.splitlines()):
                    a = res.splitlines()[idx + 1].strip()
                    cards.append((line, a))
                else:
                    cards.append((line, "A: (no answer generated)"))
        elif line.startswith("- Q:") or line.startswith("-"):
            # try cleaning bullets
            cleaned = line.lstrip("- ").strip()
            if "A:" in cleaned:
                p = cleaned.split("A:")
                cards.append(("Q: " + p[0].replace("Q:", "").strip(), "A: " + p[1].strip()))
    # fallback: if parsing fails, try simple splitting by "Q:"
    if not cards:
        parts = res.split("Q:")
        for p in parts[1:]:
            if "A:" in p:
                q, a = p.split("A:", 1)
                cards.append(("Q: " + q.strip(), "A: " + a.strip()))
    # limit to n
    return cards[:n]


def clean_text_for_pdf(text, width=100):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    text = re.sub(r'\s+', ' ', text)            # remove multiple spaces
    wrapped = "\n".join(textwrap.wrap(text, width))
    return wrapped

def save_pdf(filename, summary, transcript, flashcards, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Lecture Notes", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    # Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, clean_text_for_pdf(summary))

    # Transcript
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Transcript", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    safe_transcript = transcript[:3000] + ("\n\n...truncated..." if len(transcript) > 3000 else "")
    pdf.multi_cell(0, 6, clean_text_for_pdf(safe_transcript))

    # Flashcards
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Flashcards", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for i, (q, a) in enumerate(flashcards):
        pdf.multi_cell(0, 6, f"Q{i+1}: {clean_text_for_pdf(q)}\nA: {clean_text_for_pdf(a)}\n")

    pdf.output(pdf_path)


### Streamlit UI ###
st.set_page_config(page_title="Voice-to-Notes", layout="wide")
st.title("🎙 Lecture Voice → Notes Converter")
st.write("Upload an audio file (mp3/wav/m4a). We transcribe locally, summarize, and create flashcards.")

with st.expander("Models & status"):
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if not st.session_state.models_loaded:
        st.info("Loading models (this runs only once). Be patient — models will download on first run.")
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
    with st.spinner("Saving audio..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/mp3")
    if st.button("Transcribe & Generate Notes"):
        with st.spinner("Transcribing..."):
            segments, transcript = transcribe_with_faster_whisper(st.session_state.whisper_model, tmp_path)
        st.subheader("Transcript (click to expand)")
        st.expander("Full Transcript", expanded=False).write(transcript)

        st.subheader("Segments")
        for s in segments[:10]:
            st.write(f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text'][:200]}")

        # Summarize
        with st.spinner("Summarizing (hierarchical)..."):
            summary, chunk_summaries = hierarchical_summarize(st.session_state.summarizer, transcript)
        st.subheader("Summary")
        st.write(summary)

        st.subheader("Chunk summaries (for long lectures)")
        for i, cs in enumerate(chunk_summaries):
            st.markdown(f"**Chunk {i+1}**: {cs}")

        # Key points (extract top sentences by naive splitting)
        st.subheader("Key points (extractive, naive)")
        sentences = [s.strip() for s in transcript.split(".") if len(s.strip())>20]
        # show first 8 sentences as naive keypoints
        for i, s in enumerate(sentences[:8]):
            st.write(f"- {s[:300]}")

        # Flashcards
        with st.spinner("Generating flashcards..."):
            flashcards = generate_flashcards(st.session_state.flash_gen, summary if len(summary)>100 else transcript, n=8)
        st.subheader("Flashcards")
        for q,a in flashcards:
            st.markdown(f"**{q}**\n\n{a}")

        # Download buttons
        notes_text = f"Title: {uploaded.name}\n\nSummary:\n{summary}\n\nFlashcards:\n"
        for q,a in flashcards:
            notes_text += f"{q}\n{a}\n\n"
        notes_text += "\n\nTranscript:\n" + transcript

        st.download_button("Download notes (.txt)", notes_text, file_name="lecture_notes.txt")
        pdf_path = os.path.join(tempfile.gettempdir(), "lecture_notes.pdf")
        save_pdf(uploaded.name, summary, transcript, flashcards, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download notes (.pdf)", f, file_name="lecture_notes.pdf")
