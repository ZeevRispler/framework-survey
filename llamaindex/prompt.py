# Basic LlamaIndex Prompting
import os
import openai
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

# Load API key from environment
load_dotenv()

# Necessary for LlamaIndex to work with OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.base_url = os.environ["OPENAI_BASE_URL"]

# Initialize LLM
llm = OpenAI(model="gpt-4o-mini")

def main():
    # Simple prompting
    response = llm.complete("What is machine learning?")
    print(response.text)

if __name__ == "__main__":
    main()