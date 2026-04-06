from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model(model="deepseek:deepseek-chat")

agent = create_agent(model=model)

# 第一轮
agent.invoke({"messages": [{"role": "user", "content": "我叫张三"}]})

# 第二轮 - 不记得第一轮！
response = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]})
# AI 会说"不知道"
for message in response["messages"]:
    message.pretty_print()


from langgraph.checkpoint.memory import InMemorySaver

# 1. 创建 Agent 时添加 checkpointer
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver()  # 添加Memory
)

# 2. 调用时指定 thread_id
config = {"configurable": {"thread_id": "conversation_1"}}

# 第一轮
agent.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    config=config
)

# 第二轮 - 记得第一轮！
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么？"}]},
    config=config
)
# AI 会说"你叫张三"
for message in response["messages"]:
    message.pretty_print()