from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
llm = ChatOllama(temperature=0, model="qwen3:8b")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names"],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_ouput = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_ouput

def main():
    result = chain.invoke(
        input = {
            "input": "search for 3 job summer internship postings for a data-related position in singapore on linkedin and list their details."
        }
    )
    print(result)


if __name__ == "__main__":
    main()
