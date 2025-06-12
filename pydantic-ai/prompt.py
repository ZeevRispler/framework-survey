# Basic PydanticAI Prompting

from dotenv import load_dotenv
from pydantic_ai import Agent

# Load API key from environment
load_dotenv()

# Create agent with OpenAI model
agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt='Be concise, reply with one sentence.'
)

def main():
    # Simple prompting
    result = agent.run_sync('What is machine learning?')
    print(result.output)

if __name__ == "__main__":
    main()