from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)


@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式，如：
    "10 + 5"
    "(10 + 5) * 3"
    """
    result = eval(expression)
    return str(result)


agent = create_agent(
    model=model,
    tools=[calculator],
    system_prompt="你是一个数学助手。当遇到复杂计算时，分步骤计算。"
)

print("\n问题：先算 10 加 20，然后把结果乘以 3")
response = agent.invoke({
    "messages": [{"role": "user", "content": "先算 10 加 20，然后把结果乘以 3"}]
})

# 统计工具调用次数
tool_calls_count = 0
for msg in response['messages']:
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        tool_calls_count += len(msg.tool_calls)

print(f"\n工具调用次数: {tool_calls_count}")
print(f"最终答案: {response['messages'][-1].content}")
