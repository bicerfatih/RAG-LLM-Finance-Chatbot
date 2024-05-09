import streamlit as st  # Importing the Streamlit library for building web applications
import logging  # Importing the logging module for logging messages
from dotenv import load_dotenv  # Importing load_dotenv function to load environment variables
from create_database import database_main  # Importing a module to create a database
from web_search import perform_web_search, initialize_web_search, create_web_search_chain  # Importing functions related to web search
from langchain.chat_models import ChatOpenAI  # Importing a class for ChatOpenAI from the langchain module
from langchain_community.embeddings import OpenAIEmbeddings  # Importing a class for OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory  # Importing classes related to memory management
from langchain_community.vectorstores import Chroma  # Importing a class for Chroma
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo  # Importing classes and functions related to agent toolkits
from langchain.evaluation import load_evaluator  # Importing a function to load an evaluator
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)  # Importing classes from the langchain.schema

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv('./.env')

# Initialize global components (web search, language model, memory, and vector stores)
search_api, embeddings = initialize_web_search()
web_chain = create_web_search_chain(search_api, embeddings)
llm = ChatOpenAI(model="gpt-4-turbo", max_tokens=1024, streaming=True, verbose=True)

# Setup database and memory storage
history = FileChatMessageHistory('chat_history.json')  # Creating a FileChatMessageHistory object to manage chat history, storing it in 'chat_history.json'
llm_memory = ConversationBufferMemory(memory_key='chat_history', chat_memory=history, return_messages=True)  # Creating a ConversationBufferMemory object to manage conversation memory using the chat history
embedding_function = OpenAIEmbeddings()  # Creating an instance of OpenAIEmbeddings for embedding functions
store = Chroma(persist_directory='./chroma', embedding_function=embedding_function, collection_name='financial_documents')  # Creating a Chroma vector store with specified parameters
vectorstore_info = VectorStoreInfo(name="pdf_chat_data", description="financial_documents_pdf", vectorstore=store)  # Creating VectorStoreInfo to provide information about the vector store
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)  # Creating a VectorStoreToolkit to manage vector stores and provide functionality to the agent
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True, memory=llm_memory, agent_executor_kwargs={"handle_parsing_errors": True})  # Creating a vector store agent to execute queries, utilizing language model and toolkit
evaluator = load_evaluator("trajectory", llm=llm)  # Loading an evaluator for evaluating agent performance - trajectory predictions

# Function to determine if a question is finance-related
def is_finance_related(question):
    """Determine if the question is finance related."""
    finance_prompt = f"Question: '{question}'\n Is this question related to finance? Respond with 'yes' or 'no'."
    is_finance_response = llm.invoke(finance_prompt)  # Invoking the language model with the finance prompt
    is_finance_answer = is_finance_response.content.strip().lower()  # Stripping and converting to lowercase
    logging.info(f"Finance check for question '{question}': {is_finance_answer}")  # Logging the finance check result
    return is_finance_answer == 'yes'  # Returning True if the answer is 'yes', False otherwise

# Function to display relevant document content if available
def relevant_local_document_content(prompt):
    """Display relevant document content if available."""
    with st.expander('Relevant Document Content'):  # Creating an expander widget to expand/collapse content
        try:
            results = store.similarity_search_with_score(prompt)  # Performing a similarity search with the provided prompt
            if results:  # Checking if results are available
                st.write("Document Information: ", results[0][0].metadata)  # Displaying document information
                st.write("Content: ", results[0][0].page_content)  # Displaying document content
            else:  # If no results are available
                st.write("No relevant documents found.")  # Displaying a message indicating no relevant documents found
        except Exception as e:  # Handling exceptions
            st.error(f"Failed to retrieve document details: {str(e)}")  # Displaying an error message if document details retrieval fails

# Function to generate a response from the agent based on the input prompt
def agent_response(prompt):
    """Generate a response from the agent based on the input prompt."""
    if is_finance_related(prompt):  # Checking if the prompt is finance-related
        local_response = agent_executor.invoke(prompt)  # Invoking the agent executor with the prompt
        eval_result = evaluator.evaluate_agent_trajectory(  # Evaluating the agent trajectory
            input=local_response['input'],
            prediction=local_response['output'],
            agent_trajectory=local_response['output'],
        )
        print(eval_result['score'])  # Printing the evaluation score
        if eval_result['score'] > 0.5:  # Checking if the evaluation score is greater than 0.5
            relevant_local_document_content(prompt[0].content)  # Displaying relevant document content if available
            return local_response['output']  # Returning the agent's output
        else:  # If the evaluation score is not high enough
            web_url = perform_web_search(prompt, web_chain)  # Performing a web search
            if web_url:  # If web search results are available
                return f"No information available in our documents.\n\n More information can be found here: {web_url['sources']} \n\n {web_url['answer']}"  # Returning a message with web search results
            else:  # If no web search results are available
                return "No information available in our documents. No information available online."  # Returning a message indicating no information available online
    else:  # If the prompt is not finance-related
        return "Sorry, I am only allowed to reply to finance-related questions."  # Returning a message indicating that only finance-related questions are allowed

# Streamlit GUI setup
def setup_ui():
    # Setting up the user interface using Streamlit
    st.title("LLM Chat App - Finance")  # Setting the title of the application
    with st.sidebar:
        st.markdown("### Domain: Finance\n#### Focus: Company reports available in PDF format\n#### Fallback: Web search")  # Adding information to the sidebar about the domain and focus of the application
        st_extra = ('''
                Extra: Optional System Role    
                E.g.: Type "Include the current Euro to Dollar exchange rate in your response." as a system message.
                ''')
        st.markdown(st_extra)  # Displaying an optional system message instruction in the sidebar
        system_message = st.text_input("Optional System Message:")  # Text input for optional system message
        if st.button("Submit", key='system_message_submit_button'):  # Button to submit the system message
            st.session_state.messages.append(SystemMessage(content=system_message))  # Appending the system message to the session state

    prompt = st.text_area("Please ask your question here:")  # Text area for user input question
    if st.button("Submit", key='query_submit_button'):  # Button to submit the user question
        st.session_state.messages.append(
            HumanMessage(content=prompt),  # Appending the user question to the session state as a HumanMessage
        )
        response = agent_response(st.session_state.messages)  # Generating response from the agent based on the user question
        st.write(response)  # Displaying the response
        st.session_state.messages = []  # Resetting the session state messages

if __name__ == "__main__":

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []  # Initializing session state messages if not already present
        database_main()  # Calling the main function to generate the data store

    setup_ui()  # Calling the setup_ui function to set up the Streamlit user interface
