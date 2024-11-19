import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_vector_store():
    # Get the current directory of the script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(current_directory, "vector_store")
    
    # Initialize embeddings
    embedding_model = OllamaEmbeddings(model="llama3.1")
    
    # Load the vector store
    vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_store

def create_rag_chain(vector_store):
    # Initialize the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize the language model
    llm = ChatOllama(model="llama3.1")
    
    # Create a custom prompt template
    template = """
    Use ONLY the context provided to answer the prompt.
    Context: {context}
    Prompt: {question}
    
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    # Load the vector store
    vector_store = load_vector_store()
    
    # Create the RAG chain
    rag_chain = create_rag_chain(vector_store)
    
    # Interactive query loop
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
        
        # Perform the query
        try:
            response = rag_chain.invoke(query)
            print("\nResponse:", response)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()