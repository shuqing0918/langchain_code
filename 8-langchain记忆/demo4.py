from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
load_dotenv()
from langchain.agents import create_agent

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)


@tool
def get_order_status(order_id: str) -> str:
    """查询订单状态"""
    orders = {
        "12345": "已发货，预计明天送达",
        "67890": "配送中，今天下午送达"
    }
    return orders.get(order_id, "订单不存在")


# 客户今天上午咨询
with SqliteSaver.from_conn_string("D:/GP/sqdata/checkpoints.sqlite") as checkpointer:
    agent = create_agent(model=model, tools=[get_order_status], checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "zhang"}}
    agent.invoke({"messages": [{"role": "user", "content": "订单 12345 在哪？"}]}, config)

# 下午客户再次咨询（即使服务重启）
with SqliteSaver.from_conn_string("D:/GP/sqdata/checkpoints.sqlite") as checkpointer:
    agent = create_agent(model=model, tools=[get_order_status], checkpointer=checkpointer)
    response = agent.invoke({"messages": [{"role": "user", "content": "到了吗？"}]}, config)
    # Agent 记得上午查询的订单号！
    for message in response["messages"]:
        message.pretty_print()