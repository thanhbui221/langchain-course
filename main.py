from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from schemas import AgentResponse
from langchain.messages import HumanMessage, AIMessage, SystemMessage

tools = [TavilySearch()]
llm = ChatOllama(temperature=0, model="qwen2.5:7b")


agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)

def main():
    # human_msg = HumanMessage("search for 3 job summer internship postings for a data-related position in singapore on linkedin and list their details")
    # messages = [human_msg]
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
    print(result)


if __name__ == "__main__":
    main()
