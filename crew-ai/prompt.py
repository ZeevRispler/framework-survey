# Basic CrewAI Single Agent

from dotenv import load_dotenv
from crewai import Agent, Task, Crew

# Load API key from environment
load_dotenv()


def main():
    # Create a simple agent
    writer = Agent(
        role='Content Writer',
        goal='Write clear and engaging content',
        backstory='You are an experienced writer who creates concise, helpful content.',
        verbose=True,
        allow_delegation=False
    )

    # Create a simple task
    writing_task = Task(
        description='Write a brief explanation about machine learning in simple terms.',
        expected_output='A 2-3 sentence explanation of machine learning that anyone can understand.',
        agent=writer
    )

    # Create and run the crew
    crew = Crew(
        agents=[writer],
        tasks=[writing_task],
        verbose=True
    )

    result = crew.kickoff()
    print("\n" + "=" * 50)
    print("RESULT:")
    print(result)


if __name__ == "__main__":
    main()