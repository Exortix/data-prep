import os
import tempfile
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.embeddings import OllamaEmbeddings

# Initialize Ollama embeddings
embedding_model = OllamaEmbeddings(model="ollama3.1")

input_text = input("Enter Course Name: ")
post_fix = f"{input_text}-vector-db-ollama"

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

    # Save the vector store to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store.save_local(temp_dir)
        print(f"Vector store created and saved locally in temporary directory: {temp_dir}")

# Call the function to create the vector store
create_vector_db()
