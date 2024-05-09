"""
In this script, I'll evaluate the performance of the LLM model that I use against another model from Hugging Face.

I'll compare their classification results to assess their effectiveness."""

import logging  # Importing the logging module for logging messages
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline  # Importing necessary modules from the transformers library
from langchain.chat_models import ChatOpenAI  # Importing custom ChatOpenAI model
from dotenv import load_dotenv  # Importing the load_dotenv function to load environment variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv('./.env')

# Function to initialize the custom LLM model

def initialize_ChatOpenAI_model():
    return ChatOpenAI(model="gpt-4-turbo", max_tokens=1024)

# Function to initialize Hugging Face model and tokenizer
def initialize_hugging_face_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier

# Function to classify questions using the custom LLM model
def classify_with_custom_model(llm, question):
    prompt_message = {
        "role": "system",
        "content": f"Question: '{question}'\nIs this question related to finance? Respond with 'yes' or 'no'."
    }
    response = llm.invoke([prompt_message])
    return response.content.strip().lower() if hasattr(response, "content") else None

# Function to classify questions using the Hugging Face model
def classify_with_hf_model(classifier, question):
    hf_response = classifier(question)
    sentiment_label = hf_response[0]['label']
    return "yes" if sentiment_label == "POSITIVE" else "no"

# Function to evaluate accuracy of classification models
def evaluate_models(test_cases, llm, hf_classifier):
    llm_correct = hf_correct = 0

    for question, expected in test_cases:
        try:
            custom_result = classify_with_custom_model(llm, question)
            hf_result = classify_with_hf_model(hf_classifier, question)

            llm_correct += custom_result == expected
            hf_correct += hf_result == expected

            logging.info(f"Question: '{question}' | Expected: {expected}")
            logging.info(f"Custom Model: {custom_result} | Hugging Face Model: {hf_result}")
            logging.info(f"Custom Pass: {custom_result == expected} | Hugging Face Pass: {hf_result == expected}\n")

        except Exception as e:
            logging.error(f"Error processing question: '{question}' | Error: {e}")

    total_questions = len(test_cases)
    llm_accuracy = (llm_correct / total_questions) * 100
    hf_accuracy = (hf_correct / total_questions) * 100

    logging.info(f"Custom Model Accuracy: {llm_accuracy:.2f}%")
    logging.info(f"Hugging Face Model Accuracy: {hf_accuracy:.2f}%")

if __name__ == "__main__":
    # Define test cases
    test_cases = [
        # Clearly finance-related questions
        ("What is the current interest rate?", "yes"),
        ("How do I invest in mutual funds?", "yes"),
        ("Can you explain what a 401(k) is?", "yes"),
        ("What are the current tax implications for investing in stocks?", "yes"),
        ("What's the difference between a traditional IRA and a Roth IRA?", "yes"),
        # Non-finance questions
        ("What's the weather like today?", "no"),
        ("What is the best way to bake a chocolate cake?", "no"),
        ("Tell me about the best travel destinations for 2024.", "no"),
        ("How do I grow tomatoes in my garden?", "no"),
        ("What is the history of the French Revolution?", "no"),
        # Ambiguous questions (checking robustness)
        ("What is the best way to save?", "yes"),
        ("How do I ensure financial security?", "yes"),
        ("What are the key skills needed for software development?", "no"),
        ("Explain the concept of budgeting.", "yes"),
        ("How does insurance work?", "yes")
    ]

    # Initialize models
    llm_model = initialize_ChatOpenAI_model()
    hf_classifier = initialize_hugging_face_model("FinanceInc/auditor_sentiment_finetuned")

    # Evaluate models
    evaluate_models(test_cases, llm_model, hf_classifier)
