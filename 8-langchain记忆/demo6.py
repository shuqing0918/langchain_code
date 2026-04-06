from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
from langchain_core.tools import tool

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)
db_path = "tools.sqlite"


@tool
def get_order_status(order_id: str) -> str:
    """查询订单状态"""
    orders = {
        "12345": "已发货，预计明天送达",
        "67890": "配送中，今天下午送达"
    }
    return orders.get(order_id, "订单不存在")


with SqliteSaver.from_conn_string(db_path) as checkpointer:
    agent = create_agent(
        model=model,
        tools=[get_order_status],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=checkpointer
    )

    config = {"configurable": {"thread_id": "customer_001"}}

    print("\n第一轮：查询订单")
    print("客户: 查询订单 12345 的状态")
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "查询订单 12345 的状态"}]},
        config=config
    )
    print(f"Agent: {response1['messages'][-1].content}")

    print("\n第二轮：询问之前的查询结果")
    print("客户: 我的订单什么时候到？")
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我的订单什么时候到？"}]},
        config=config
    )
    print(f"Agent: {response2['messages'][-1].content}")
