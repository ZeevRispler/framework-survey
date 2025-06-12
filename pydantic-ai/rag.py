# PydanticAI RAG with Tool

from dotenv import load_dotenv
from pydantic_ai import Agent

# Load API key from environment
load_dotenv()

# Mock knowledge base
knowledge_base = {
    "python": "Python is a high-level programming language known for its simplicity and readability.",
    "machine_learning": "Machine Learning is a subset of AI that enables computers to learn from data.",
    "pydantic": "Pydantic is a data validation library that uses Python type hints."
}

# Create agent with search tool
agent = Agent('openai:gpt-4o-mini')

@agent.tool_plain
def search_knowledge(query: str) -> str:
    """Search the knowledge base for information about a topic."""
    query_lower = query.lower().replace(" ", "_")
    for key, value in knowledge_base.items():
        if key in query_lower or query_lower in key:
            return value
    return "No information found about this topic."

def main():
    # Query with RAG capability
    result = agent.run_sync('Tell me about Python programming language')
    print(result.output)

if __name__ == "__main__":
    main()