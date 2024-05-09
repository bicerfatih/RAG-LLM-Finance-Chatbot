import logging  # Importing the logging module for logging messages
from dotenv import load_dotenv  # Importing load_dotenv function to load environment variables
from langchain.chat_models import ChatOpenAI  # Importing ChatOpenAI class from langchain.chat_models module
from langchain_community.vectorstores import Chroma  # Importing Chroma class from langchain_community.vectorstores module
from langchain_community.embeddings import OpenAIEmbeddings  # Importing OpenAIEmbeddings class from langchain_community.embeddings module
from langchain.retrievers.web_research import WebResearchRetriever  # Importing WebResearchRetriever class from langchain.retrievers.web_research module
from langchain.chains import RetrievalQAWithSourcesChain  # Importing RetrievalQAWithSourcesChain class from langchain.chains module
from langchain.memory import FileChatMessageHistory, ConversationSummaryBufferMemory  # Importing classes from langchain.memory module
from langchain_community.utilities import GoogleSearchAPIWrapper  # Importing GoogleSearchAPIWrapper class from langchain_community.utilities module

# Setup logging
logging.basicConfig(level=logging.INFO)  # Configuring logging to show INFO level messages
logger = logging.getLogger(__name__)  # Creating a logger instance with the current module name

# Load environment variables
load_dotenv('./.env')  # Loading environment variables from .env file

def initialize_web_search():
    """Initialize the components required for the web search."""
    logger.info("Initializing the web search components.")  # Logging an INFO message indicating initialization
    search_api = GoogleSearchAPIWrapper()  # Creating an instance of GoogleSearchAPIWrapper for web search API
    embeddings = OpenAIEmbeddings()  # Creating an instance of OpenAIEmbeddings for embeddings
    return search_api, embeddings  # Returning initialized search_api and embeddings components

def create_web_search_chain(search_api, embeddings):
    """Create the web search chain with the given components."""
    logger.info("Creating the web search chain.")  # Logging an INFO message indicating creation of web search chain
    model = ChatOpenAI(model="gpt-4-turbo", max_tokens=1024, streaming=True, verbose=True)  # Creating an instance of ChatOpenAI model
    vector_store = Chroma(embedding_function=embeddings, persist_directory='./chroma_db_oai')  # Creating a Chroma vector store
    retriever = WebResearchRetriever.from_llm(vectorstore=vector_store, llm=model, search=search_api)  # Creating a retriever for web research
    history = FileChatMessageHistory('web_chat_history.json')  # Creating a file-based chat message history
    memory = ConversationSummaryBufferMemory(llm=model, input_key='question', output_key='answer', memory=history, return_messages=True)  # Creating a conversation memory
    web_chain = RetrievalQAWithSourcesChain.from_chain_type(model, retriever=retriever, memory=memory)  # Creating a retrieval QA chain with sources
    return web_chain  # Returning the created web search chain

def perform_web_search(query, web_chain):
    """Perform a web search for the given query using the specified web chain."""
    try:
        logger.info(f"Performing web search for the query: {query}")  # Logging an INFO message indicating the query being searched
        result = web_chain({'question': query})  # Performing the web search using the provided query
        logger.info("Web search completed successfully.")  # Logging an INFO message indicating successful completion of web search
        return result  # Returning the search result
    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")  # Logging an ERROR message if an exception occurs during web search
        return None  # Returning None if an error occurs during web search

def main():
    """Main function to handle the web search."""
    query = ''  # Initializing the query variable
    search_api, embeddings = initialize_web_search()  # Initializing web search components
    web_chain = create_web_search_chain(search_api, embeddings)  # Creating web search chain
    result = perform_web_search(query, web_chain)  # Performing web search
    if result:
        logger.info(f"Search Results: {result}")  # Logging search results if available
    else:
        logger.info("No results returned.")  # Logging if no results are returned

if __name__ == "__main__":
    main()  # Calling the main function if the script is executed directly

