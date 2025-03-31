import streamlit as st
import torch
import assemblyai as aai
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# Set up AssemblyAI API Key
aai.settings.api_key = "YOUR_ASSEMBLYAI_API_KEY"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="Enhancing-Search-Engine-Relavance-for-Video-Subtitles/Data/En_seach_engine_subtitles.db")
collection = chroma_client.get_or_create_collection(name="chromadb_En_sub_embeddings")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda' if torch.cuda.is_available() else 'cpu')

def transcribe_audio(file_path):
    """Upload audio and transcribe it using AssemblyAI."""
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    return transcript.text

def search_subtitles(query_text):
    """Convert query to embedding and search in ChromaDB."""
    query_embedding = embedding_model.encode([query_text])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    formatted_results = []
    if results and "documents" in results and results["documents"]:
        for doc in results["documents"]:
            formatted_results.append(f"### {doc['metadata']['name']}\n📜 **Excerpt:**\n*\"{doc['metadata']['content']}...\"*\n\n---")
    return "\n".join(formatted_results) if formatted_results else "No results found."

st.title("Subtitle Search Engine")

option = st.radio("Choose Input Method", ["Upload Audio", "Voice Search", "Text Search"])

if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_audio_path = temp_file.name
        st.audio(temp_audio_path, format='audio/wav')
        with st.spinner("Transcribing audio with AssemblyAI..."):
            text_query = transcribe_audio(temp_audio_path)
        st.success("Transcription completed!")
        st.write("### Transcribed Text:", text_query)
        search_results = search_subtitles(text_query)
        st.markdown(search_results, unsafe_allow_html=True)

elif option == "Voice Search":
    st.subheader("Record your voice and search")
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    
    if webrtc_ctx.audio_receiver:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            while True:
                try:
                    frame = webrtc_ctx.audio_receiver.get_frames(timeout=1)[0]
                    temp_audio.write(frame.to_ndarray().tobytes())
                except:
                    break
            temp_audio_path = temp_audio.name
        st.audio(temp_audio_path, format='audio/wav')
        with st.spinner("Transcribing audio with AssemblyAI..."):
            text_query = transcribe_audio(temp_audio_path)
        st.success("Transcription completed!")
        st.write("### Transcribed Text:", text_query)
        search_results = search_subtitles(text_query)
        st.markdown(search_results, unsafe_allow_html=True)

elif option == "Text Search":
    text_query = st.text_input("Enter your search query")
    if st.button("Search"):
        search_results = search_subtitles(text_query)
        st.markdown(search_results, unsafe_allow_html=True)