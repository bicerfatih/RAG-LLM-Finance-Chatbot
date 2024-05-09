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
DATA_PATHS = ['./pdf_chat_data/amazon/Morningstar_AMZN_CompanyReport_20230203.pdf',  # List of paths to PDF documents
              './pdf_chat_data/amazon/S&P_RatingsDirect_Amazon.comInc._2991372_May-24-2023.pdf',
              './pdf_chat_data/amazon/amzn-20221231.pdf']

def database_main():
    generate_data_store()  # Calling the function to generate the data store

def generate_data_store():
    documents = load_documents()  # Loading documents from specified paths
    chunks = split_text(documents)  # Splitting the loaded documents into text chunks
    save_to_chroma(chunks)  # Saving the text chunks to Chroma vector store

def load_documents():
    documents = []  # Initializing an empty list to store documents
    for path in DATA_PATHS:  # Iterating over each data path
        try:
            loader = PyPDFLoader(path)  # Creating a PyPDFLoader instance for loading PDF documents
            pages = loader.load()  # Loading pages from the PDF document
            documents.extend(pages)  # Extending the list of documents with the loaded pages
        except Exception as e:
            logging.error(f"Failed to load documents from {path}: {str(e)}")  # Logging an error if document loading fails
    return documents  # Returning the list of loaded documents

def split_text(documents: list[Document]):
    try:
        # Creating a RecursiveCharacterTextSplitter instance for splitting text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Setting the chunk size
            chunk_overlap=200,  # Setting the chunk overlap
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
