import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import whisper
import torch
import re

# Configure Gemini API
genai.configure(api_key="Replace with your actual API key")

# Load Whisper Model (Fastest on GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=device)

# Load Gemini Model
def generate_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

# Extract YouTube Video ID
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

# Get YouTube Transcript (Multilingual Support)
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Check if English transcript is available, else use auto-generated one
        if "en" in transcript_list:
            transcript = transcript_list.find_transcript(["en"])
        else:
            transcript = transcript_list.find_transcript(["hi", "es", "fr", "de"]).fetch()  # Example fallback languages

        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Whisper Transcription (For More Accuracy)
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Split Large Transcripts
def chunk_text(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

# Summarize Transcript using Gemini
def summarize_transcript(transcript):
    summary_prompt = f"Summarize this video transcript accurately in key points:\n\n{transcript}"
    return generate_gemini_response(summary_prompt)

# Convert Text to Embeddings & Store in FAISS
def create_faiss_db(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  
    docs = [Document(page_content=chunk) for chunk in chunks]
    vector_db = FAISS.from_documents(docs, embedding_model)
    return vector_db

# Answer User Questions with Memory
def answer_query(query, faiss_db, memory):
    retrieved_docs = faiss_db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    conversation_history = memory.buffer or ""
    
    prompt = f"Conversation history:\n{conversation_history}\n\nBased on this video transcript:\n{context}\n\nAnswer the question: {query}"
    
    response = generate_gemini_response(prompt)
    
    # Store question-answer in memory
    memory.save_context({"question": query}, {"answer": response})
    
    return response

# Streamlit UI
st.title("📺 Advanced YouTube Video Summarizer & Q&A")
st.write("Summarize long YouTube videos & ask any questions about them!")

# Initialize session state for storing Q&A
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

# LangChain Memory
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory()

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Get Summary"):
    if "youtube.com" in video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            with st.spinner("Fetching and summarizing..."):
                transcript = get_transcript(video_id)
                if "Error" in transcript:
                    st.error(transcript)
                else:
                    chunks = chunk_text(transcript)
                    summary = summarize_transcript(" ".join(chunks[:5]))  # Summarize first 5 chunks
                    faiss_db = create_faiss_db(chunks)

                    st.subheader("📌 Summary:")
                    st.write(summary)

                    # Store FAISS DB and Summary in Session
                    st.session_state["faiss_db"] = faiss_db
                    st.session_state["summary"] = summary
        else:
            st.error("Invalid YouTube URL. Please enter a valid link.")
    else:
        st.error("Invalid YouTube URL. Please enter a valid link.")

# Q&A Section
if "faiss_db" in st.session_state:
    user_query = st.text_input("Ask a question about the video:")
    if user_query:
        with st.spinner("Generating response..."):
            response = answer_query(user_query, st.session_state["faiss_db"], st.session_state["memory"])
            
            # Store Q&A history
            st.session_state["qa_history"].append({"question": user_query, "answer": response})

            # Display Q&A history
            st.subheader("💡 Q&A History:")
            for qa in st.session_state["qa_history"]:
                st.write(f"**Q:** {qa['question']}")
                st.write(f"**A:** {qa['answer']}")
                st.write("---")
