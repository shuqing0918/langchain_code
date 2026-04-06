from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"关于 {query} 的结果..."


model = init_chat_model(model="deepseek:deepseek-chat")

agent = create_agent(
    model=model,
    tools=[search],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "session_1"}}

# 第一轮：使用工具
agent.invoke({"messages": [{"role": "user", "content": "搜索 Python"}]}, config)
# Agent 调用 search("Python")

# 第二轮：引用之前的结果
response = agent.invoke(
    {"messages": [{"role": "user", "content": "刚才搜索的结果是什么？"}]},
    config
)
# Agent 记得工具返回的结果，无需重新调用
for message in response["messages"]:
    message.pretty_print()
