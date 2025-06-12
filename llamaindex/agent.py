# LlamaIndex Agent with Tool
import os
import openai
from dotenv import load_dotenv
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

# Load API key from environment
load_dotenv()

# Necessary for LlamaIndex to work with OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.base_url = os.environ["OPENAI_BASE_URL"]

# Define a simple tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

# Create agent with tool
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful calculator assistant.",
)

async def main():
    response = await agent.run("What is 123 * 456?")
    print(response)

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())