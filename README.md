## RAG-LLM-Finance-Chatbot
The LLM-Finance-Chatbot is designed to answer finance-related queries using company reports in PDF format.Retrieval-Augmented Generation (RAG) is used for optimising the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response.. This project showcases coding abilities and proficiency with Large Language Models (LLMs), specifically focusing on the use of GPT models for extracting and serving information from structured documents.

## Features
- **Finance Query Recognition**: Determines if a user's question is related to finance.
- **Chat Interface**: Users can interact through a simple text-based or GUI-based interface.
- **PDF Information Extraction**: Extracts and responds with information directly from financial reports.
- **Web Search Fallback**: If information is not found in the local database, the system performs a web search to find the answer.
- **Response Evaluation**: Includes a mechanism to evaluate the accuracy and relevance of the chatbot's responses.

## Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/LLM-Finance-Chatbot.git
   ```
2. Navigate to the project directory:
   ```
   cd LLM-Finance-Chatbot
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage
To run the chatbot and view this Streamlit app on a browser, execute the following command in the terminal:
```
  command:

  streamlit run src/main.py
```
Follow the on-screen prompts to interact with the chatbot.

## Documentation
Financial reports documents are available in the `pdf_chat_data/` directory. 

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request to propose your changes.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
- Thanks to the open-source community for support and contributions.
- Special thanks to [OpenAI](https://openai.com/) for providing access to GPT models.

