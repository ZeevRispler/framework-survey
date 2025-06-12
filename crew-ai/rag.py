# CrewAI with RagTool

from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import RagTool

# Load API key from environment
load_dotenv()


def main():
    # Create a mock document for RAG
    # In practice, you would point this to actual files or URLs
    rag_tool = RagTool(
        description="Search through knowledge base about programming and AI topics.",
        # You can add documents here or configure data sources
    )

    # Create researcher agent with RAG tool
    researcher = Agent(
        role='Knowledge Researcher',
        goal='Find relevant information using the knowledge base',
        backstory='You are an expert researcher who can search through documents and knowledge bases to find accurate information.',
        tools=[rag_tool],
        verbose=True,
        allow_delegation=False
    )

    # Research task using RAG
    research_task = Task(
        description='Use the RAG tool to find information about Python programming language and its applications.',
        expected_output='A comprehensive summary of Python programming language based on the knowledge base.',
        agent=researcher
    )

    # Create and run crew
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )

    result = crew.kickoff()
    print("\n" + "=" * 50)
    print("RESULT:")
    print(result)


if __name__ == "__main__":
    main()