# OpenAI Agents Platform - Main Components

## Agents
LLMs equipped with specific instructions and tools for performing tasks.

**Agents:** [docs](https://openai.github.io/openai-agents-python/)

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful coding assistant"
)

result = Runner.run_sync(agent, "Explain Python functions")
print(result.final_output)
```

## Tools
Python functions that agents can call to perform specific actions or retrieve information.

**Tools:** [docs](https://openai.github.io/openai-agents-python/)

```python
from agents.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather for a location"""
    return f"Sunny, 72°F in {location}"

agent = Agent(
    name="Weather Bot",
    instructions="Help with weather queries",
    tools=[get_weather]
)
```

## Handoffs
Mechanism for agents to delegate tasks to other specialized agents.

**Handoffs:** [docs](https://openai.github.io/openai-agents-python/)

```python
from agents import Agent, handoff

math_agent = Agent(name="Math Expert", instructions="Solve math problems")

@tool
def transfer_to_math():
    """Transfer to math specialist"""
    return handoff(math_agent)

general_agent = Agent(
    name="General Assistant",
    tools=[transfer_to_math]
)
```

## Guardrails
Input validation and safety checks that run in parallel with agent execution.

**Guardrails:** [docs](https://openai.github.io/openai-agents-python/)

```python
from agents import Agent, guardrail

@guardrail
def validate_input(message: str) -> bool:
    """Check if input is appropriate"""
    return len(message) < 1000 and not any(word in message.lower() for word in ["spam", "hack"])

agent = Agent(
    name="Safe Assistant",
    instructions="Help users safely",
    guardrails=[validate_input]
)
```

## Runner
Execution engine that manages the agent loop and handles tool calls.

**Runner:** [docs](https://openai.github.io/openai-agents-python/)

```python
from agents import Agent, Runner

agent = Agent(name="Helper", instructions="Be helpful")

# Synchronous execution
result = Runner.run_sync(agent, "Hello world")

# Asynchronous execution
async def run_async():
    result = await Runner.run(agent, "Hello world")
    return result
```