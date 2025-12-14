from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama

from schemas import AgentResponse

tools = [TavilySearchResults(max_results=5)]
# llm = ChatOpenAI(model="gpt-4o")
llm = ChatOllama(temperature=0, model="llama3.1")


agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job summer internship postings for a data-related position in singapore on linkedin and list their details",
                }
            ]
        }
    )
    # Access structured response from the agent
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
