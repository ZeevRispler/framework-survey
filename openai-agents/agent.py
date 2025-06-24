"""
OpenAI Agents SDK Example
"""

import asyncio
import json
from typing_extensions import TypedDict, Any
from agents import Agent, FunctionTool, Runner, function_tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Location(TypedDict):
    lat: float
    long: float

@function_tool
async def fetch_weather(location: Location) -> str:
    """Fetch the weather for a given location.
    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return "sunny"

agent = Agent(
    name="Assistant",
    tools=[fetch_weather],
)

for tool in agent.tools:
    if isinstance(tool, FunctionTool):
        print(tool.name)
        print(tool.description)
        print(json.dumps(tool.params_json_schema, indent=2))
        print()

async def run_agent(agent: Agent, input: str, max_turns: int = 5):
    """Run the agent with the given input."""
    result = await Runner.run(
        agent,
        input=input,
        max_turns=max_turns,
    )
    return result.final_output

if __name__ == "__main__":
    result = asyncio.run(run_agent(agent=agent, input="What is the weather in San Francisco?", max_turns=3))
    print("\n" + "─" * 50)
    print(result)