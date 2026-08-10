# YouTube RAG Chatbot

A simple YouTube Retrieval-Augmented Generation (RAG) chatbot that fetches a video's transcript, indexes it with FAISS embeddings, and answers user questions using OpenAI.

## Project Files

- `YoutubeChatbot_RAG.py` - Console-based app. Enter a YouTube video ID, index the transcript, then ask questions.
- `YoutubeChatbot_RAG_Streamlit.py` - Streamlit web app. Load a video transcript in the browser and ask questions interactively.
- `requirement.txt` - Python dependencies required to run the project.

## Features

- Fetches English transcripts from YouTube via `youtube-transcript-api`.
- Splits transcripts into chunks with `langchain` text splitters.
- Builds FAISS vector embeddings using OpenAI embedding models.
- Uses `ChatOpenAI` from `langchain_openai` for question-answering.
- Supports both CLI and Streamlit web app modes.

## Requirements

- Python 3.11+ (the environment in this repo uses Python 3.12)
- An OpenAI API key

## Setup

1. Create and activate your Python virtual environment.

   ```powershell
   python -m venv myenv
   .\myenv\Scripts\activate
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirement.txt
   ```

3. Create a `.env` file in the repository root with your OpenAI key:

   ```text
   OPENAI_API_KEY=your-openai-api-key
   ```

## Usage

### Run the console app

```powershell
python YoutubeChatbot_RAG.py
```

Enter a YouTube video ID (the 11-character string after `v=` in the URL) and ask questions in the terminal.

### Run the Streamlit app

```powershell
streamlit run YoutubeChatbot_RAG_Streamlit.py
```

Open the URL shown by Streamlit in your browser, enter a YouTube video ID, load the transcript, and start asking questions.

## Notes

- The project expects an English transcript. If the transcript is missing, disabled, or unavailable, it will show an error.
- The Streamlit app caches the vector store for the loaded video to avoid rebuilding it on every interaction.
- The OpenAI model used is `gpt-3.5-turbo` and the embedding model is `text-embedding-3-small`.

## Troubleshooting

- If `OPENAI_API_KEY` is not detected, confirm `.env` is in the project root and contains the correct key.
- If `streamlit` is missing, install it with:

  ```powershell
  pip install streamlit
  ```

- If transcript fetching fails, verify the video ID is valid and that the video has an English transcript available.
