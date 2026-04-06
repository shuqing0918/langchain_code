from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)

# 创建持久化 checkpointer（使用 with 语句）
# with SqliteSaver.from_conn_string("D:/GP/sqdata/checkpoints.sqlite") as checkpointer:
#     agent = create_agent(
#         model=model,
#         tools=[],
#         system_prompt="你是一个有帮助的助手。",
#         checkpointer=checkpointer  # 使用 SQLite
#     )
#     config = {"configurable": {"thread_id": "user_123"}}
#     # 第一次运行
#     agent.invoke({"messages": [{"role": "user", "content": "我的订单号是12345"}]}, config)

config = {"configurable": {"thread_id": "user_123"}}
# 程序重启后，对话仍然保留！
with SqliteSaver.from_conn_string("D:/GP/sqdata/checkpoints.sqlite") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer,system_prompt="你是一个有帮助的助手。",)
    response = agent.invoke({"messages": [{"role": "user", "content": "我的订单号是多少？"}]}, config)

for message in response["messages"]:
    message.pretty_print()