from config import settings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser



def get_model():
    """
    Initializes and returns the ChatOpenAI model.

    Returns:
        ChatOpenAI: The initialized ChatOpenAI model.
    """
    return ChatOpenAI(
        base_url=settings.BASE_URL,
        temperature=0,
        api_key=settings.API_KEY,
        model_name=settings.MODEL
    )


def get_prompt():
    """
    Creates and returns a chat prompt template.

    Returns:
        ChatPromptTemplate: The constructed chat prompt template.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know or that you cannot answer. Use three sentences maximum and keep the answer concise.""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", """Context:
            ---
            {context}
            ---
            {question}"""),
        ]
    )


def get_rag_chain(retriever, memory_buffer):
    """
    Constructs and returns a RAG (Retrieval-Augmented Generation) chain.

    Args:
        retriever: The retriever object for fetching relevant documents.
        memory_buffer: The memory buffer for storing chat history.

    Returns:
        RunnablePassthrough: The constructed RAG chain.
    """
    model = get_model()
    prompt = get_prompt()
    memory = RunnablePassthrough.assign(chat_history=RunnableLambda(memory_buffer.load_memory_variables) | itemgetter("chat_history"))
    
    return (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | memory
        | prompt
        | model
        | StrOutputParser()
    )
