from langchain_core.messages import trim_messages

# 模拟一个长对话历史
from langchain_core.messages import HumanMessage, AIMessage

messages = [
    HumanMessage(content="消息 1"),
    AIMessage(content="回复 1"),
    HumanMessage(content="消息 2"),
    AIMessage(content="回复 2"),
    HumanMessage(content="消息 3"),
    AIMessage(content="回复 3"),
    HumanMessage(content="消息 4"),
    AIMessage(content="回复 4"),
]

print(f"\n原始消息数: {len(messages)}")


# 按 token 数裁剪（不严格条数）	max_tokens=N + 合理 token_counter
# 严格保留最后 N 条消息	max_count=N

trimmed = trim_messages(
    messages,
    max_tokens=2,  # 或使用 token 数限制
    strategy="last",  # 保留最后的消息
    token_counter=len  # 简单计数器（实际应该用 token 计数）
)

print(f"修剪后消息数: {len(trimmed)}")
print("\n保留的消息：")
for msg in trimmed:
    # msg.pretty_print()
    print(f"  {msg.__class__.__name__}: {msg.content}")