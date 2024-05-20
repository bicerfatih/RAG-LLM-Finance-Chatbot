import streamlit as st  # Importing the Streamlit library for building web applications
import logging  # Importing the logging module for logging messages
from dotenv import load_dotenv  # Importing load_dotenv function to load environment variables
from create_database import database_main  # Importing a module to create a database
from web_search import perform_web_search, initialize_web_search, create_web_search_chain  # Importing functions related to web search
from langchain.chat_models import ChatOpenAI  # Importing a class for ChatOpenAI from the langchain module
from langchain_community.embeddings import OpenAIEmbeddings  # Importing a class for OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory  # Importing classes related to memory management
from langchain_community.vectorstores import Chroma  # Importing a class for Chroma
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo  # Importing classes and functions related to agent toolkits
from langchain.evaluation import load_evaluator  # Importing a function to load an evaluator
from langchain.schema import SystemMessage, HumanMessage, AIMessage  # Importing classes from the langchain.schema

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv('./.env')

# Initialize global components (web search, language model, memory, and vector stores)
search_api, embeddings = initialize_web_search()
web_chain = create_web_search_chain(search_api, embeddings)
llm = ChatOpenAI(model="gpt-4-turbo", max_tokens=1024, streaming=True, verbose=True)

# Setup database and memory storage
history = FileChatMessageHistory('chat_history.json')
llm_memory = ConversationBufferMemory(memory_key='chat_history', chat_memory=history, return_messages=True)
embedding_function = OpenAIEmbeddings()
store = Chroma(persist_directory='./chroma', embedding_function=embedding_function, collection_name='financial_documents')
vectorstore_info = VectorStoreInfo(name="pdf_chat_data", description="financial_documents_pdf", vectorstore=store)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True, memory=llm_memory, agent_executor_kwargs={"handle_parsing_errors": True}, max_iterations=10)
#evaluator = load_evaluator("trajectory", llm=llm)

# Function to determine if a question is finance-related
def is_finance_related(question):
    """Determine if the question is finance related."""
    finance_prompt = f"Question: '{question}'\nIs this question related to finance? Respond with 'yes' or 'no'."
    is_finance_response = llm.invoke(finance_prompt)
    is_finance_answer = is_finance_response.content.strip().lower()
    logging.info(f"Finance check for question '{question}': {is_finance_answer}")
    return is_finance_answer == 'yes'

# Function to display relevant document content if available
def relevant_local_document_content(prompt):
    """Display relevant document content if available."""
    with st.expander('Relevant Document Content'):
        try:
            results = store.similarity_search_with_score(prompt) # Also, store.similarity_search_with_relevance_score(prompt, k=1)
            if results:
                st.write("Document Information: ", results[0][0].metadata)
                st.write("Content: ", results[0][0].page_content)
                st.write("similarity score: ", results[0][1])  # Similarity score
                print(results[0][1])  # Similarity score
            else:
                st.write("No relevant documents found.")
        except Exception as e:
            st.error(f"Failed to retrieve document details: {str(e)}")

# Function to generate a response from the agent based on the input prompt
def agent_response(prompt):
    """Generate a response from the agent based on the input prompt."""
    if is_finance_related(prompt):
        # First try to find a relevant document from the local vector store
        results = store.similarity_search_with_score(prompt)
        if results[0][1] < 0.3:
            relevant_local_document_content(prompt)
            agent_local_response = agent_executor.invoke(prompt)
            print(agent_executor.invoke(prompt))
            return agent_local_response['output']

        # If no relevant document is found, fall back to web search
        web_url = perform_web_search(prompt, web_chain)
        print("There is no relevant information in our documents. Moving to web search.")
        if web_url:
            return f"No information available in our documents.\n\nMore information can be found here: {web_url['sources']}\n\n{web_url['answer']}"
        else:
            return "No information available in our documents. No information available online."
    else:
        return "Sorry, I am only allowed to reply to finance-related questions."

# Streamlit GUI setup
def setup_ui():
    st.title("LLM Chat App - Finance")
    with st.sidebar:
        st.markdown("### Domain: Finance\n#### Focus: Company reports available in PDF format\n#### Fallback: Web search")
        st_extra = ('''
                Extra: Optional System Role    
                E.g.: Type "Include the current Euro to Dollar exchange rate in your response." as a system message.
                ''')
        st.markdown(st_extra)
        system_message = st.text_input("Optional System Message:")
        if st.button("Submit", key='system_message_submit_button'):
            st.session_state.messages.append(SystemMessage(content=system_message))

    prompt = st.text_area("Please ask your question here:")
    if st.button("Submit", key='query_submit_button'):
        st.session_state.messages.append(HumanMessage(content=prompt))
        response = agent_response(prompt)
        st.write(response)
        st.session_state.messages = []

if __name__ == "__main__":
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
        database_main()
    setup_ui()
