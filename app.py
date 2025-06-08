import streamlit as st
import fitz  # PyMuPDF
import speech_recognition as sr
import whisper
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile

# Load Whisper model
asr_model = whisper.load_model("base")

st.title("Voice + PDF Q&A App")

# Upload PDF
pdf_file = st.file_uploader("C:\Users\Iraj Qureshi\streamlit\venv\sample_database.pdf", type="pdf")
if pdf_file is not None:
    st.success("PDF uploaded successfully!")

# Optional text question
question = st.text_input("Ask a question:")

# Upload audio
audio_file = st.file_uploader("C:\Users\Iraj Qureshi\streamlit\venv\WhatsApp Audio 2025-06-05 at 15.31.30_ac56b143.waptt.opus", type=["wav", "mp3", "m4a"])

if pdf_file:
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline("text-generation", model="gpt2")),
        retriever=retriever
    )

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        result = asr_model.transcribe(tmp_path)
        st.success(f"Transcribed Question: {result['text']}")
        answer = qa_chain.run(result['text'])
        st.write("Answer:", answer)

    elif question:
        answer = qa_chain.run(question)
        st.write("Answer:", answer)
