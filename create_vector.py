import os
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain_ollama import OllamaEmbeddings  # Updated import
from langchain.docstore.document import Document

# Initialize Ollama embeddings
embedding_model = OllamaEmbeddings(model="llama3")

def create_vector_db():
    # Specify the directory containing the PDF files and convert it to an absolute path
    pdf_directory = os.path.abspath('./pdfs')
    documents = []
    
    # Iterate through all files in the pdfs directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            # Add pages to documents list with metadata
            documents.extend([
                Document(page_content=page.page_content, metadata={"source": filename})
                for page in pages
            ])

    # Create a FAISS vector store
    vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)

    # Save the vector store to the same directory as the Python file
    current_directory = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(current_directory, "vector_store")

    # Save the vector store locally
    vector_store.save_local(vector_store_path)
    print(f"Vector store created and saved locally in: {vector_store_path}")

# Call the function to create the vector store
if __name__ == "__main__":
    input_text = input("Enter Course Name: ")
    create_vector_db()
