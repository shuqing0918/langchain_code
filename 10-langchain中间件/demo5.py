from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# 1. 初始化模型
model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1
)


# 2. 定义一个假的 send_email 工具
@tool
def send_email(to: str, subject: str, content: str) -> str:
    """
    发送邮件（示例工具，不会真的发送）
    """
    return (
        "【模拟发送成功】\n"
        f"收件人: {to}\n"
        f"主题: {subject}\n"
        f"内容: {content}"
    )



# 3. 创建带 Human-in-the-loop 的 Agent
agent = create_agent(
    model=model,
    tools=[send_email],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True  # 在调用 send_email 前中断
            }
        )
    ]
)

# 必须指定 thread_id，否则无法 resume
config = {"configurable": {"thread_id": "hitl_demo"}}


# 4. 触发一个会调用 send_email 的请求

print("用户发起请求...")
response = agent.invoke(
    {"messages": [{
        "role": "user",
        "content": "帮我给 test@example.com 发一封邮件，主题是测试，内容是你好"
    }]},
    config=config
)

# print(response)

# 5. 此时流程已被中断（不会真正执行工具）
print("已触发 Human-in-the-loop，中断在工具调用前")
print("当前 Agent 输出：\n")
print(response["messages"][-1].content)


# 6. 模拟人工审批
choice = input("是否同意发送邮件？(y/n): ").strip().lower()

if choice != "y":
    print("人工拒绝了该操作")
else:
    print("人工同意，继续执行\n")

    last_msg = response["messages"][-1]
    tool_call = last_msg.tool_calls[0]
    tool_result = send_email.invoke(tool_call["args"])
    result = agent.invoke(
        {"messages": [
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                }
            ]},
        config=config
    )

    print("Agent 最终回复：\n")
    print(result["messages"][-1].content)
