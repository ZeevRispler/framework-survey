# PydanticAI - Components

## 🤖 Agents

Define type-safe agents with custom dependencies and structured outputs.

**Class**: `Agent[DepsT, OutputT]`  
**Base Class**: Generic Agent class  
**Docs**: [Agents](https://ai.pydantic.dev/agents/)

```python
from pydantic_ai import Agent

# Custom typed agent
agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="You are a specialized assistant."
)
```

## 🛠️ Tools

Create functions that agents can call, with or without access to dependencies.

**Decorators**: `@agent.tool`, `@agent.tool_plain`  
**Docs**: [Tools](https://ai.pydantic.dev/tools/)

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o-mini')

# Tool with context access
@agent.tool
def search_database(ctx: RunContext, query: str) -> str:
    """Search the database for information."""
    # Access dependencies via ctx.deps
    return f"Database results for: {query}"

# Simple tool without context
@agent.tool_plain  
def calculate(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)
```

## 📝 System Prompts

Create dynamic prompts that can access runtime context and dependencies.

**Decorator**: `@agent.system_prompt`  
**Docs**: [System Prompts](https://ai.pydantic.dev/agents/#system-prompts)

```python
from pydantic_ai import Agent, RunContext
from datetime import datetime

agent = Agent('openai:gpt-4o-mini', deps_type=str)

@agent.system_prompt
def dynamic_prompt(ctx: RunContext[str]) -> str:
    username = ctx.deps
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    You are assisting {username}.
    Current time: {current_time}
    Be helpful and personalized.
    """
```

## ✅ Output Validators

Add validation logic to ensure outputs meet specific requirements.

**Decorator**: `@agent.output_validator`  
**Docs**: [Output Validation](https://ai.pydantic.dev/output/#output-validator-functions)

```python
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: int  # 1-10 scale

agent = Agent('openai:gpt-4o-mini', output_type=Response)

@agent.output_validator
def validate_confidence(ctx: RunContext, output: Response) -> Response:
    if output.confidence < 1 or output.confidence > 10:
        raise ModelRetry("Confidence must be between 1-10")
    return output
```

## 🔧 Models

Integrate custom LLM providers or modify model behavior.

**Class**: `Model`  
**Base Class**: `Model` abstract class  
**Docs**: [Custom Models](https://ai.pydantic.dev/models/)

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel

openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
response = agent.run_sync('What is the capital of France?')
```

