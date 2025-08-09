
# CrewAI with OpenRouter LLM integration
import os
from dotenv import load_dotenv
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

load_dotenv()

OPENROUTER_API_KEY = "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"
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "https://example.com")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "Example Site")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# Custom CrewAI tool for OpenRouter LLM

class OpenRouterLLMTool(BaseTool):
    name: str = "LLM Response Tool"
    description: str = "Send a message to the LLM and get a response using OpenRouter."

    def _run(self, message: str) -> str:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            extra_body={},
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content

llm_tool = OpenRouterLLMTool()

# Create two agents that can communicate
researcher = Agent(
    role="Research Specialist",
    goal="Conduct thorough research on given topics",
    backstory="You're an expert researcher with access to various sources",
    allow_delegation=True,
    verbose=True,
    tools=[llm_tool]
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging content based on research",
    backstory="You're a skilled writer who transforms research into compelling content",
    allow_delegation=True,
    verbose=True,
    tools=[llm_tool]
)

# Create collaborative tasks
research_task = Task(
    description="Research the latest trends in AI agents for 2025",
    expected_output="Comprehensive research summary with key findings",
    agent=researcher
)

writing_task = Task(
    description="Write an engaging article based on the research findings",
    expected_output="Well-structured 500-word article about AI agent trends",
    agent=writer,
    context=[research_task]
)

# Create crew with collaborative agents
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True
)

# Run the application
result = crew.kickoff()
print(result)
