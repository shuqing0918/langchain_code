from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain.agents.middleware import (
    PIIMiddleware,
)

load_dotenv()


model = ChatDeepSeek(model="deepseek-chat")


# 定义一个简单的工具函数
@tool("query_order_status", description="根据订单号查询订单状态")
def query_order_status(order_id: str) -> str:
    if order_id == "1024":
        return "订单 1024 的状态是：已发货，预计送达时间是 3-5 个工作日。"
    else:
        return f"未找到订单 {order_id} 的信息，请检查订单号是否正确。"


# 创建智能体
agent = create_agent(
    # model="openai:gpt-4o",
    model=model,
    tools=[query_order_status],
    middleware=[
        # 脱敏中间件
        PIIMiddleware(
            pii_type="email",
            strategy="redact",  # 表示检测到PII后替换为REDACTED_EMAIL
            apply_to_input=True,  # 表示处理用户输入**，在发送给LLM前脱敏
            # pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            detector=r"\b[\w\.-]+@qq\.com\b"
        )
    ],
    system_prompt="你是一个专业的客服助手，帮助用户查询订单状态。当用户提到订单号时，直接调用query_order_status工具查询订单状态，不要询问用户确认。"
)

# 测试运行
if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": 
    # "帮我查询1024订单状态"
    "我的邮箱是123@qq.com，帮我查询1024订单状态"
    }]})
    print("AI回复：", result)
    # print(result['messages'][-1].content)
    for message in result["messages"]:
        message.pretty_print()
    # print(result["messages"].pretty_print())