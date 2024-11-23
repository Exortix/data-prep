import os
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain_community.document_loaders import TextLoader  # For .txt and .md files
from langchain_community.document_loaders import CSVLoader  # For .csv files
from langchain_community.document_loaders import JSONLoader  # For .json files
from langchain_ollama import OllamaEmbeddings  # Updated import
from langchain.docstore.document import Document

# Initialize Ollama embeddings
embedding_model = OllamaEmbeddings(model="llama3.1")

def load_documents_from_file(file_path, extension):
    """
    Load documents from a file based on its extension.
    """
    documents = []
    
    if extension == '.pdf':
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        documents.extend(
            pages
        )
    elif extension in ['.txt', '.md']:
        loader = TextLoader(file_path)
        content = loader.load()
        documents.extend(content)
    elif extension == '.csv':
        loader = CSVLoader(file_path)
        rows = loader.load()
        documents.extend(
            rows
        )
    elif extension == '.json':
        # Example jq_schema: extracting "content" field from a nested structure
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".data[].content",  # Modify this as per your JSON schema
        )
        content = loader.load()
        documents.extend(
        content
        )

    return documents

def create_vector_db():
    # Specify the directory containing the files and convert it to an absolute path
    data_directory = os.path.abspath('./data')
    documents = []

    # Iterate through all files in the data directory
    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        extension = os.path.splitext(filename)[1].lower()
        
        # Load documents based on their extensions
        if extension in ['.pdf', '.txt', '.md', '.csv', '.json']:
            documents.extend(load_documents_from_file(file_path, extension))

    # Create a FAISS vector store
    vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)

    # Save the vector store to the same directory as the Python file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(current_directory, "multi_data_vector_store")

    # Save the vector store locally
    vector_store.save_local(vector_store_path)
    print(f"Vector store created and saved locally in: {vector_store_path}")

# Call the function to create the vector store
if __name__ == "__main__":
    create_vector_db()



'''
JSON format should look like this

{
    "data": [
        {"content": "Nuclear physics explores the properties and behavior of atomic nuclei."},
        {"content": "Applications of nuclear physics include energy generation and medical imaging."},
        {"content": "Research in nuclear physics contributes to understanding the universe at its most fundamental level."}
    ]
}
'''