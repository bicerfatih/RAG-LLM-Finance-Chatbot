"""
In this script, I'll evaluate the faithfulness, answer_relevancy, context_recall, context_precision

and answer_correctness of the RAG response"""

import create_database  # Importing a custom module for database creation
from langchain_community.vectorstores import Chroma  # Importing Chroma from a community module
from langchain.chat_models import ChatOpenAI  # Importing ChatOpenAI model
from langchain_community.embeddings import OpenAIEmbeddings  # Importing OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, \
    VectorStoreInfo  # Importing tools for creating a vector store agent
from ragas import evaluate  # Importing evaluate function from ragas library
from ragas.metrics import (  # Importing specific metrics for evaluation
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from dotenv import load_dotenv  # Importing load_dotenv function to load environment variables
from datasets import Dataset, Features, Value, Sequence  # Importing necessary classes from datasets module
import pandas as pd  # Importing pandas for data manipulation and analysis

# Load environment variables from the .env file
load_dotenv('./.env')

# Initialize OpenAI LLM (Language Model) with GPT-4 Turbo configuration
llm = ChatOpenAI(model="gpt-4-turbo", max_tokens=1024)

# Initialize OpenAIEmbeddings for the embedding function
embedding_function = OpenAIEmbeddings()

# Initialize the Chroma vector store with the specified parameters
store = Chroma(persist_directory='./chroma', embedding_function=embedding_function,
               collection_name='financial_documents')

# Create a retriever from the Chroma vector store
retriever = store.as_retriever()

# Create a VectorStoreToolkit using vector store information and OpenAI LLM
vectorstore_info = VectorStoreInfo(
    name="pdf_chat_data",
    description="financial_documents_pdf",
    vectorstore=store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)

# Create an agent executor using the OpenAI LLM and toolkit, with verbose logging
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True,
                                          agent_executor_kwargs={"handle_parsing_errors": True})

# Define features for the dataset
features = Features({
    'question': Value('string'),
    'answer': Value('string'),
    'contexts': Sequence(Value('string')),
    'ground_truth': Value('string')
})

# Define test questions and ground truths
questions = [" What are the key risks associated with Amazon's online retail segment?",
             "What percentage of total U.S. retail e-commerce sales did Amazon's sales represent in 2022",
             "How has Amazon's advertising business grown in terms of percentage of total sales from 2019 to 2022?",
             "What are Amazon's expectations for AWS sales growth in the years 2024 and 2025?",
             "What percentage of global IT spending is currently in the cloud, according to Amazon"]

ground_truths = [["The key risks include volatile profitability due to costs related to fulfillment network expansion, technology, and marketing which constrain margins."],
                 ["Amazon's U.S. retail e-commerce sales represented almost 15% of total retail sales."],
                 ["Amazon's advertising sales grew from 4.5% of total sales in 2019 to 7.3% in 2022."],
                 ["Amazon forecasts AWS sales growth to accelerate in 2024 and 2025, expecting it to be in the 10% to 15% area."],
                 ["Just 5-10% of global IT spending is in the cloud."]]

answers = []
contexts = []

# Generate answers and retrieve relevant contexts for each question
for query in questions:
    answers.append(agent_executor.invoke(query))

    relevant_docs = retriever.get_relevant_documents(query)
    if relevant_docs and len(relevant_docs) > 0:
        contexts.append([docs.page_content for docs in relevant_docs])
    else:
        contexts.append("No context found")  # Fallback if no context is retrieved

# Construct dataset dictionary
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}

# Create a dataset from the dictionary with defined features
dataset = Dataset.from_dict(data, features=features)

# Score the generated answers against the expected answers using specified metrics
result = evaluate(dataset=dataset,
                  metrics=[faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness])

# Convert the evaluation result to a pandas DataFrame for better visualization
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
df = result.to_pandas()
print(df)
