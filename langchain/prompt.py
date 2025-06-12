# Minimal LangChain LLM Prompting Example

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Method 1: Simple string prompt
response = model.invoke("What is machine learning?")
print(response.content)

# Method 2: Using prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{question}")
])

if __name__ == "__main__":
    # Example usage with a question
    chain = prompt_template | model
    result = chain.invoke({"question": "Explain Python in one sentence"})
    print(result.content)