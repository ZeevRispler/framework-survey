# CrewAI - Available Components

CrewAI provides built-in components for creating collaborative multi-agent systems. Each component is designed for role-based agent coordination and workflow management.

## 🤖 Agent Configuration

Create agents with distinct roles, goals, and personalities for specialized tasks.

**Available**: Role-based agents with customizable behavior and delegation  
**Docs**: [Agents](https://docs.crewai.com/concepts/agents)

```python
from crewai import Agent

# Pre-configured agent roles
researcher = Agent(
    role='Senior Data Researcher',
    goal='Uncover cutting-edge developments in AI',
    backstory='Expert researcher with deep knowledge of AI trends',
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

analyst = Agent(
    role='Data Analyst', 
    goal='Analyze data and provide insights',
    backstory='Experienced analyst skilled at finding patterns',
    verbose=True,
    allow_delegation=True
)
```

## 📋 Task Management

Configure tasks with dependencies, context sharing, and structured outputs.

**Available**: Sequential tasks, parallel execution, context passing, file outputs  
**Docs**: [Tasks](https://docs.crewai.com/concepts/tasks)

```python
from crewai import Task

research_task = Task(
    description='Research latest AI developments in 2025',
    expected_output='List of 10 key AI developments',
    agent=researcher,
    tools=[search_tool],
    output_file='research_results.md'
)

analysis_task = Task(
    description='Analyze the research findings',
    expected_output='Detailed analysis report',
    agent=analyst,
    context=[research_task],  # Depends on research task
    output_json=True
)
```

## 🎭 Crew Orchestration

Configure different workflow processes for agent collaboration.

**Available**: Sequential, hierarchical, and custom process types  
**Docs**: [Crews](https://docs.crewai.com/concepts/crews)

```python
from crewai import Crew, Process

# Sequential workflow
sequential_crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True,
    memory=True
)

# Hierarchical workflow with manager
hierarchical_crew = Crew(
    agents=[manager, researcher, analyst],
    tasks=[planning_task, research_task, analysis_task],
    process=Process.hierarchical,
    manager_llm=manager_llm,
    verbose=True
)
```

## 🛠️ Pre-built Tools

Use ready-made tools for common tasks like web search, file operations, and data processing.

**Available**: 40+ tools for web scraping, file handling, search, APIs  
**Docs**: [Tools](https://docs.crewai.com/concepts/tools)

```python
from crewai_tools import (
    SerperDevTool,      # Web search
    ScrapeWebsiteTool,  # Web scraping
    FileReadTool,       # File reading
    PDFSearchTool,      # PDF search
    CodeInterpreterTool # Code execution
)

# Configure tools with parameters
search_tool = SerperDevTool(n_results=10)
scrape_tool = ScrapeWebsiteTool()
pdf_tool = PDFSearchTool(pdf_path="document.pdf")

# Assign tools to agents
researcher = Agent(
    role='Researcher',
    tools=[search_tool, scrape_tool, pdf_tool]
)
```





