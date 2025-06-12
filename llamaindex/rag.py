# LlamaIndex RAG Example
import os
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load API key from environment
load_dotenv()

# Necessary for LlamaIndex to work with OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.base_url = os.environ["OPENAI_BASE_URL"]

def main():
    # Create mock documents instead of reading from files
    from llama_index.core import Document

    mock_documents = [
        Document(
            text="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."),
        Document(
            text="Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."),
        Document(
            text="LlamaIndex is a data framework designed to help developers build applications with large language models by providing tools for data ingestion, indexing, and querying.")
    ]

    # Create index from mock documents
    index = VectorStoreIndex.from_documents(mock_documents)

    # Create query engine
    query_engine = index.as_query_engine()

    # Query the documents
    response = query_engine.query("What is Python programming language?")
    print(response)


if __name__ == "__main__":
    main()