import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import MergerRetriever

def load_vector_stores():
    # Get the current directory of the script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    vector_stores_path = os.path.join(current_directory, "vector_stores")
    
    # Initialize embeddings
    embedding_model = OllamaEmbeddings(model="llama3.1")
    
    retrievers = []
    
    # Load all vector stores in the directory and create retrievers
    for folder_name in os.listdir(vector_stores_path):
        folder_path = os.path.join(vector_stores_path, folder_name)
        
        if os.path.isdir(folder_path):
            # Load the vector store
            vector_store = FAISS.load_local(folder_path, embedding_model, allow_dangerous_deserialization=True)
            
            # Create a retriever from the vector store
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            retrievers.append(retriever)
    
    return retrievers

def create_rag_chain(retrievers):
    # Merge the retrievers using MergerRetriever
    merged_retriever = MergerRetriever(retrievers=retrievers)
    
    # Initialize the language model
    llm = ChatOllama(model="llama3.1", temperature=0.2)
    
    # Create a custom prompt template
    template = """
    Use ONLY the context provided to generate.
    Context: {context}
    Prompt: {question}
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Create the RAG chain
    rag_chain = (
        {"context": merged_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    # Load the retrievers from vector stores
    retrievers = load_vector_stores()
    
    # Create the RAG chain with merged retrievers
    rag_chain = create_rag_chain(retrievers)
    
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
