from utils import load_documents, get_embeddings, get_memory, get_retriever
from models import get_rag_chain, get_model

def stream_query(query, history, rag_chain):
    """
    Streams the query to the RAG chain and prints the response.

    Args:
        query (str): The user's query.
        history: The chat history object.
        rag_chain: The RAG chain object.

    Returns:
        None
    """
    history.add_user_message(query)
    response = ""
    for answer in rag_chain.stream(query):
        response += answer
        print(answer, end="", flush=True)
    history.add_ai_message(response)

def main():
    """
    Main function to initialize the model, memory, retriever, and start the query loop.

    Returns:
        None
    """
    model = get_model()
    memory_buffer, history = get_memory(model)
    
    documents = load_documents()
    embeddings = get_embeddings()
    retriever = get_retriever(embeddings, documents)
    
    rag_chain = get_rag_chain(retriever, memory_buffer)
    
    try:
        while True:
            message = input("Send a message: ")
            stream_query(message, history, rag_chain)
            print("\n")
    except EOFError:
        print("\nEOF received. Exiting the program.")
    except KeyboardInterrupt:
        print("\nInterrupt received. Exiting the program.")

if __name__ == "__main__":
    main()
