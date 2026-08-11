# This will accept the user input video__id and the Query and generates the answer for the Query

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

parser = StrOutputParser()

video_id = input("Enter the video id: ")
Query = input("Enter your query : ")
try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=["en"])
    transcript = " ".join(snippet.text for snippet in fetched_transcript)
except TranscriptsDisabled:
    print("Transcript is disabled for this video.")

# indexing--->Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Indexing--->Embedding
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# define vector store
vector_store = FAISS.from_documents(chunks, embeddings)

# Define Retriever
retriever = vector_store.as_retriever(
    search_type='similarity', search_kwargs={'k': 4})

model = ChatOpenAI()

prompt = PromptTemplate(
    template="""you are a helpful assitant,
    answer only from the provided transcript context.
    if the context is insufficient, just say you don't know.
    {context}
    question:{question}""",
    input_variables=['context', 'question']
)

# this will have top 4 results in doc format,
# retriever_docs = retriever.invoke(Query)
# retriever_docs.page_content will have the data

# Now contactinate all the docs to generate the final context

# context_text = "\n\n".join(doc.page_content for doc in retriever_docs)
# for the above we will write the function


def format_docs(retrived_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
    return context_text


parallelChain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
)

sequentialChain = RunnableSequence(prompt, model, parser)

finalChain = RunnableSequence(parallelChain, sequentialChain)


result = finalChain.invoke(Query)

print(result)
