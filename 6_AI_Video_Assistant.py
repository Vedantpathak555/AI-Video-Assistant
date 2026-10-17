import os, tempfile, uuid
import chromadb
import yt_dlp
import static_ffmpeg
from dotenv import load_dotenv
from groq import Groq
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os, tempfile, uuid, subprocess
import streamlit as st
from fpdf import FPDF


# ---------- Part 1: Setup ----------
load_dotenv(override=True)
static_ffmpeg.add_paths()  # ffmpeg + ffprobe PATH me jod deta hai
model = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0)
stt = Groq()  # GROQ_API_KEY .env se leta hai

def ask(prompt, system="You are a helpful meeting assistant."):
    return model.invoke([("system", system), ("human", prompt)]).content

# ---------- Part 2: Audio nikalna ----------
def get_audio(url):
    d = tempfile.mkdtemp()
    opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{d}/a.%(ext)s",
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "32"}],
        "postprocessor_args": {"ffmpeg": ["-ac", "1", "-ar", "16000"]},
    }
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])
    return f"{d}/a.mp3"

def save_upload(f):  # Streamlit ki uploaded file ke liye
    p = os.path.join(tempfile.mkdtemp(), f.name)
    with open(p, "wb") as out:
        out.write(f.getbuffer())
    return p

# ---------- Part 3: Transcription ----------
def transcribe(path, lang=None):  # lang="hi" Hindi ke liye, None = auto-detect
    try:
        with open(path, "rb") as f:
            r = stt.audio.transcriptions.create(
                file=(os.path.basename(path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                language=lang)
        return r if isinstance(r, str) else r.text
    finally:
        os.remove(path)  # audio file transcription ke baad delete

# ---------- Part 4: Clean English translation ----------
def to_english(text, size=6000):
    sys = ("Translate the meeting transcript into clean, natural English. "
           "Fix filler words and broken sentences, keep names, numbers and meaning exact. "
           "If it is already English, just clean it. Output only the text, nothing else.")
    parts = [text[i:i + size] for i in range(0, len(text), size)]
    return "\n".join(ask(p, sys) for p in parts)


# ---------- Part 5: Summary ----------
def summarize(eng):
    sys = "You summarize meetings. Be concise and factual. Do not invent anything."
    return ask(f"Summarize this meeting in 5-8 bullet points:\n\n{eng}", sys)

# ---------- Part 6: Action items, decisions, questions ----------
def extract(eng):
    sys = "You extract facts from meetings. Use only what is in the transcript. Do not invent anything."
    tasks = {
        "Action Items": "List action items as bullets in the format: Task | Owner | Deadline. Write 'Not mentioned' where owner or deadline is missing.",
        "Key Decisions": "List the key decisions made in this meeting as bullets.",
        "Open Questions": "List open questions and follow-ups that remain unresolved, as bullets.",
    }
    return {k: ask(f"{v} If there are none, write 'None'.\n\nTranscript:\n{eng}", sys) for k, v in tasks.items()}

# ---------- Part 7: Chunks + ChromaDB (in-memory) ----------
chroma = chromadb.EphemeralClient()  # sirf RAM me, disk par kuch nahi

def build_store(eng, size=800, overlap=100):
    chunks = [eng[i:i + size] for i in range(0, len(eng), size - overlap)]
    col = chroma.create_collection("m" + uuid.uuid4().hex[:8])
    col.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
    return col

# ---------- Part 8: RAG chat ----------
def chat(col, question, k=4):
    hits = col.query(query_texts=[question], n_results=min(k, col.count()))["documents"][0]
    context = "\n---\n".join(hits)
    sys = ("Answer the question using ONLY the meeting context given. "
           "If the answer is not in the context, say 'This was not discussed in the meeting.'")
    return ask(f"Context:\n{context}\n\nQuestion: {question}", sys)

# ---------- Part 9: Export ----------
def make_report(d):
    return "\n\n".join(f"## {k}\n{v}" for k, v in d.items())

def make_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, report.encode("latin-1", "replace").decode("latin-1"))
    return bytes(pdf.output())

def compress(p):  # upload ki badi video/audio ko chhote mp3 me badalta hai (Groq limit ke liye)
    out = p.rsplit(".", 1)[0] + "_c.mp3"
    subprocess.run(["ffmpeg", "-y", "-i", p, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", out],
                   check=True, capture_output=True)
    os.remove(p)
    return out

# ---------- Part 10: Streamlit UI ----------
st.set_page_config(page_title="AI Video Assistant")
st.title("🎙️ AI Video Assistant")

src = st.radio("Input", ["YouTube URL", "Upload file"], horizontal=True)
lang = {"Auto": None, "Hindi/Hinglish": "hi", "English": "en"}[
    st.selectbox("Language", ["Auto", "Hindi/Hinglish", "English"])]
url = st.text_input("YouTube URL") if src == "YouTube URL" else ""
up = st.file_uploader("Audio/Video", type=["mp3", "mp4", "wav", "m4a", "webm"]) if src == "Upload file" else None

if st.button("Process"):
    if not (url or up):
        st.warning("Pehle URL ya file do.")
        st.stop()
    try:
        with st.spinner("Audio nikal raha hoon..."):
            path = get_audio(url) if src == "YouTube URL" else compress(save_upload(up))
        with st.spinner("Transcribe ho raha hai..."):
            text = transcribe(path, lang)
        with st.spinner("Analyze ho raha hai..."):
            eng = to_english(text)
            st.session_state.data = {"Summary": summarize(eng), **extract(eng)}
            st.session_state.col = build_store(eng)
            st.session_state.history = []
    except Exception as e:
        st.error(f"Error: {e}")

d = st.session_state.get("data")
if d:
    for k, v in d.items():
        st.subheader(k)
        st.markdown(v)

    report = make_report(d)
    c1, c2 = st.columns(2)
    c1.download_button("Download TXT", report, "meeting_report.txt")
    c2.download_button("Download PDF", make_pdf(report), "meeting_report.pdf")

    st.divider()
    st.subheader("Chat with meeting")
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)
    if q := st.chat_input("Meeting ke baare me kuch puchho"):
        st.chat_message("user").write(q)
        a = chat(st.session_state.col, q)
        st.chat_message("assistant").write(a)
        st.session_state.history += [("user", q), ("assistant", a)]