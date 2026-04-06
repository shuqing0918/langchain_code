from dotenv import load_dotenv
load_dotenv()
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


model = init_chat_model(model="deepseek:deepseek-chat")


agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个有帮助的助手。",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "long_conversation"}}

# 模拟多轮对话
print("\n模拟 10 轮对话...")
for i in range(1, 11):
    agent.invoke(
        {"messages": [{"role": "user", "content": f"这是第 {i} 轮对话"}]},
        config=config
    )

# 查看消息数量
response = agent.invoke(
    {"messages": [{"role": "user", "content": "总结一下"}]},
    config=config
)

print(f"\n总消息数: {len(response['messages'])}")
print("(包含用户消息 + AI 回复)")