from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import settings
import os



def load_documents():
    """
    Loads and splits documents from a PDF file.

    Returns:
        list: A list of document chunks.
    """
    loader = PyPDFLoader(settings.FILE_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return loader.load_and_split(text_splitter)


def get_embeddings():
    """
    Initializes and returns the embeddings model.

    Returns:
        OllamaEmbeddings: The embeddings model.
    """
    return OllamaEmbeddings(model=settings.EMBEDDINGS_MODEL)


def get_memory(model):
    """
    Initializes and returns the memory buffer and chat history.

    Args:
        model (ChatOpenAI): The chat model.

    Returns:
        tuple: A tuple containing the memory buffer and chat history.
    """
    history = ChatMessageHistory()
    memory_buffer = ConversationBufferWindowMemory(
        llm=model,
        return_messages=True,
        memory_key="chat_history",
        chat_memory=history
    )
    return memory_buffer, history


def get_retriever(embeddings, documents):
    """
    Initializes and returns the retriever.

    Args:
        embeddings (OllamaEmbeddings): The embeddings model.
        documents (list): The list of document chunks.

    Returns:
        FAISS: The retriever object.
    """
    if os.path.exists("vectorstore"):
        faiss_index = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    else:
        faiss_index = FAISS.from_documents(documents=documents, embedding=embeddings)
        faiss_index.save_local("vectorstore")

    return faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
