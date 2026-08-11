
"""
Setup:
    1. pip install -r requirements.txt   (make sure "streamlit" is in there)
    2. Create a .env file in this same folder with:
           OPENAI_API_KEY=your-key-here
    3. Run: streamlit run 5_YoutubeChatbot_RAG_withStreamlit.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")
st.title("YouTube RAG Chatbot")
st.caption("Ask questions about any YouTube video's transcript.")

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY not found. Add it to a .env file in this folder and restart.")
    st.stop()


def fetch_transcript(video_id: str):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=["en"])
        return " ".join(snippet.text for snippet in fetched_transcript), None
    except TranscriptsDisabled:
        return None, "Transcript is disabled for this video."
    except NoTranscriptFound:
        return None, "No English transcript found for this video."
    except VideoUnavailable:
        return None, "This video is unavailable. Double-check the video ID (must be 11 characters)."
    except Exception as e:
        return None, f"Unexpected error fetching transcript: {e}"


@st.cache_resource(show_spinner=False)
def build_vectorstore(video_id: str, transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""You are a helpful assistant.
    Answer only from the provided transcript context.
    If the context is insufficient, just say you don't know.
    {context}
    Question: {question}""",
        input_variables=["context", "question"],
    )


def answer_question(question: str, retriever, prompt: PromptTemplate, llm: ChatOpenAI) -> str:
    retrieved_docs = retriever.invoke(question)
    final_prompt = prompt.invoke(
        {"context": retrieved_docs, "question": question})
    response = llm.invoke(final_prompt)
    return response.content


# Session state setup

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer) tuples

prompt_template = build_prompt()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)


# Step 1: Video ID input + transcript loading

with st.form("video_form"):
    video_id_input = st.text_input(
        "YouTube Video ID",
        placeholder="e.g. ZDa-Z5JzLYM (the part after v= in the URL, not the full link)",
    )
    load_clicked = st.form_submit_button("Load Transcript")

if load_clicked and video_id_input:
    with st.spinner("Fetching transcript and building the index..."):
        transcript, error = fetch_transcript(video_id_input.strip())
        if error:
            st.error(error)
        else:
            vectorstore, num_chunks = build_vectorstore(
                video_id_input.strip(), transcript)
            st.session_state.vectorstore = vectorstore
            st.session_state.video_id = video_id_input.strip()
            st.session_state.chat_history = []
            st.success(
                f"Transcript loaded and indexed into {num_chunks} chunks. Ask away below!")


# Step 2: Question + answer, once a transcript is loaded

if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader(f"Ask about video: {st.session_state.video_id}")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Your question")
        ask_clicked = st.form_submit_button("Ask")

    if ask_clicked and question:
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        with st.spinner("Thinking..."):
            answer = answer_question(question, retriever, prompt_template, llm)
        st.session_state.chat_history.append((question, answer))

    # Show chat history, most recent first
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")
