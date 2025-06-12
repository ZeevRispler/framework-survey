# PydanticAI Agent with Structured Output

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Load API key from environment
load_dotenv()

# Define structured output
class CalculationResult(BaseModel):
    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Brief explanation of the calculation")

# Create agent with tool and structured output
agent = Agent(
    'openai:gpt-4o-mini',
    output_type=CalculationResult,
    system_prompt='You are a helpful calculator that explains your work.'
)

@agent.tool_plain
def calculate(expression: str) -> float:
    """Calculate a mathematical expression safely."""
    try:
        # Simple evaluation - in production use a safer parser
        result = eval(expression)
        return result
    except Exception:
        return 0.0

def main():
    # Run agent with structured output
    result = agent.run_sync('What is 25 * 37?')
    print(f"Expression: {result.output.expression}")
    print(f"Result: {result.output.result}")
    print(f"Explanation: {result.output.explanation}")

if __name__ == "__main__":
    main()