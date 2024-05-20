from langchain.document_loaders import DirectoryLoader  # Importing DirectoryLoader for loading documents from a directory
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing RecursiveCharacterTextSplitter for splitting text
from langchain.schema import Document  # Importing Document schema for defining document structure
from langchain.embeddings import OpenAIEmbeddings  # Importing OpenAIEmbeddings for generating embeddings
from langchain.vectorstores.chroma import Chroma  # Importing Chroma vector store for storing document chunks
from langchain.document_loaders import PyPDFLoader  # Importing PyPDFLoader for loading PDF documents
import os  # Importing os module for file operations
import logging  # Importing logging module for logging messages
import shutil  # Importing shutil module for file operations
from dotenv import load_dotenv  # Importing load_dotenv function to load environment variables

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Configuring logging format and level

load_dotenv('./.env')  # Loading environment variables from .env file

CHROMA_PATH = "chroma"  # Setting the directory path for Chroma vector store
root_folder = './pdf_chat_data'
def database_main():
    generate_data_store()  # Calling the function to generate the data store

def generate_data_store():
    documents = load_documents(root_folder)  # Loading documents from specified paths
    chunks = split_text(documents)  # Splitting the loaded documents into text chunks
    save_to_chroma(chunks)  # Saving the text chunks to Chroma vector store

def load_documents(root_folder):
    """
    Load all PDF documents from a specified root directory and its subdirectories.
    Returns:
        list: A list of loaded documents from all found PDF files.
    """
    documents = []
    file_count = 0  # Initialize a counter to track the number of files processed

    # Walk through all directories starting from the root_folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):  # Check if the file is a PDF
                full_path = os.path.join(dirpath, filename)  # Get the full path of the file
                try:
                    loader = PyPDFLoader(full_path)  # Create a loader for the PDF file
                    pages = loader.load()  # Load pages from the PDF file
                    documents.extend(pages)  # Add the loaded pages to the documents list
                    file_count += 1  # Increment the file counter
                    logging.info(f"Successfully processed {full_path}, total pages loaded: {len(pages)}")
                except Exception as e:
                    logging.error(
                        f"Failed to load documents from {full_path}: {str(e)}")  # Log an error if loading fails

    logging.info(f"Total PDF files processed: {file_count}")
    return documents  # Return the list of loaded documents

def split_text(documents: list[Document]):
    try:
        # Creating a RecursiveCharacterTextSplitter instance for splitting text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Setting the chunk size
            chunk_overlap=100,  # Setting the chunk overlap
            length_function=len,  # Setting the length function
            add_start_index=True,  # Adding start index to chunks
        )
        chunks = text_splitter.split_documents(documents)  # Splitting documents into chunks
        logging.info(f"Split {len(documents)} pages into {len(chunks)} chunks.")  # Logging the number of pages and chunks
        print(f"Split {len(documents)} pages into {len(chunks)} chunks.")  # Printing the number of pages and chunks
    except Exception as e:
        logging.error(f"Failed to split documents: {str(e)}")  # Logging an error if document splitting fails
        chunks = []  # Initializing an empty list of chunks
    return chunks  # Returning the list of chunks

def save_to_chroma(chunks: list[Document]):
    """Save document chunks to Chroma vector store."""
    try:
        # Ensuring the directory is clean before creating a new database
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)  # Removing the existing Chroma directory
        # Creating a Chroma vector store from the chunks
        store = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH, collection_name='financial_documents'
        )
        store.persist()  # Persisting the vector store
        logging.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")  # Logging the number of chunks saved
    except Exception as e:
        logging.error(f"Failed to save chunks to Chroma: {str(e)}")  # Logging an error if saving chunks to Chroma fails

if __name__ == "__main__":
    database_main()  # Calling the main function to generate the data store

