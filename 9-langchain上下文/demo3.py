from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model


@tool
def calculator(operation: str, a: float, b: float) -> str:
    """执行数学计算"""
    return a * b


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)

# 创建客服 Agent
agent = create_agent(
    model=model,
    tools=[calculator],
    system_prompt="""你是客服助手。
        特点：
        - 记住用户问题
        - 简洁回答
        - 使用工具计算
    """,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=300  # 适合客服场景
        )
    ]
)

config = {"configurable": {"thread_id": "customer_123"}}

# 模拟客服对话
conversations = [
    "你好，我想咨询订单",
    "我的订单号是 12345",
    "帮我算一下 100 乘以 2 的优惠价",
    "谢谢"
]

for msg in conversations:
    print(f"\n客户: {msg}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config=config
    )
    print(f"客服: {response['messages'][-1].content}")
    print("=" * 50)
