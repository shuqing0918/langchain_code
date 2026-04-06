# 1. 基础环境
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model

# 2. 定义工具（Tools）
@tool
def query_order_status(order_id: str) -> str:
    """根据订单号查询订单状态"""
    return f"订单 {order_id} 当前状态：已支付，待发货"


@tool
def query_shipping_status(order_id: str) -> str:
    """根据订单号查询物流信息"""
    return f"订单 {order_id} 的物流信息：顺丰快递，运输中"


# 3. 初始化模型
model = init_chat_model(model="deepseek:deepseek-chat")


# 4. 初始化 Checkpointer（记忆）
checkpointer = InMemorySaver()


# 5. 创建 Agent
agent = create_agent(
    model=model,
    tools=[query_order_status, query_shipping_status],
    system_prompt=(
        "你是一个电商客服助手。"
        "如果用户在对话中提供了订单号，你需要记住它。"
        "当用户询问订单状态或物流信息时，如果已经知道订单号，"
        "请直接使用工具查询；如果不知道订单号，请先向用户确认。"
    ),
    checkpointer=checkpointer
)


# 6. 对外服务函数
def customer_service(session_id, message):
    """
    session_id: 用户会话 ID（同一个 ID 共享记忆）
    message: 用户输入
    """
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": message}
            ]
        },
        config=config
    )

    return result["messages"][-1].content


# 7. 示例运行
if __name__ == "__main__":
    print(customer_service("user_001", "你好"))
    print("="*50)
    print(customer_service("user_002", "我的订单号是12345"))
    print("="*50)
    print(customer_service("user_001", "我的订单号是123456"))
    print("="*50)
    print(customer_service("user_002", "帮我查一下物流"))