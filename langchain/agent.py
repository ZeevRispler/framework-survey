# Minimal LangChain Agent with Tool

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Initialize model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define a simple tool
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression. Use this for any math problems."""
    try:
        result = eval(expression)
        return f"The answer is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":

    # Create agent with tool
    tools = [calculator]
    agent = create_react_agent(model, tools)

    # Test the agent
    response = agent.invoke({"messages": [HumanMessage(content="What is 25 * 37?")]})
    print(response["messages"][-1].content)