"""
YouTube RAG Chatbot
--------------------
Fetches a YouTube video's transcript, indexes it into a FAISS vector store,
and lets you ask questions about the video's content using an OpenAI chat model.

Setup:
    1. pip install -r requirements.txt
    2. Create a .env file in this same folder with:
           OPENAI_API_KEY=your-key-here
    3. Run: python youtube_rag_chatbot.py
"""

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
import os

# Setup: load API key from .env (never hardcode this)

from dotenv import load_dotenv
load_dotenv()


if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY not found. Create a .env file with OPENAI_API_KEY=your-key-here"
    )


def fetch_transcript(video_id: str) -> str | None:
    """Step 1a - Indexing: fetch the transcript text for a given YouTube video ID."""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=["en"])
        return " ".join(snippet.text for snippet in fetched_transcript)
    except TranscriptsDisabled:
        print("Transcript is disabled for this video.")
    except NoTranscriptFound:
        print("No English transcript found for this video.")
    except VideoUnavailable:
        print(
            "This video is unavailable. Double-check the video ID (must be 11 characters).")
    return None


def build_vectorstore(transcript: str):
    """Step 1b-1d - Indexing: split transcript into chunks and embed into FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    print(f"Split transcript into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


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
    """Step 2-4: Retrieve relevant chunks, augment the prompt, and generate an answer."""
    retrieved_docs = retriever.invoke(question)
    final_prompt = prompt.invoke(
        {"context": retrieved_docs, "question": question})
    response = llm.invoke(final_prompt)
    return response.content


def main():
    video_id = input(
        "Enter the YouTube video ID (the part after v= in the URL): ").strip()

    transcript = fetch_transcript(video_id)
    if transcript is None:
        return

    vectorstore = build_vectorstore(transcript)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4})
    prompt = build_prompt()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    print("\nTranscript indexed. Ask questions about the video (type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = answer_question(question, retriever, prompt, llm)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
