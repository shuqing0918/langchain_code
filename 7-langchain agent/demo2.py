from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)


@tool
def _get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=model,
    tools=[_get_weather],
    system_prompt="你是一个有帮助的助手。"
)

input = {"messages": [{"role": "user", "content": "北京天气怎么样？"}]}
for chunk in agent.stream(input, stream_mode="updates"):
    # print(chunk)
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
