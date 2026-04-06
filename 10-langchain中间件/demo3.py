from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent

model = init_chat_model(model="deepseek:deepseek-chat")


class MessageTrimmerMiddleware(AgentMiddleware):
    """
    消息修剪中间件 - 限制消息数量

    before_model 修改消息列表
    注意：需要配合无 checkpointer 使用，否则历史会被恢复
    """

    def __init__(self, max_messages=5):
        super().__init__()
        self.max_messages = max_messages
        self.trimmed_count = 0  # 统计修剪次数

    def before_model(self, state, runtime):
        """模型调用前，修剪消息"""
        messages = state.get('messages', [])

        if len(messages) > self.max_messages:
            # 保留最近的 N 条消息
            trimmed_messages = messages[-self.max_messages:]
            self.trimmed_count += 1
            print(f"\n[修剪] 消息从 {len(messages)} 条减少到 {len(trimmed_messages)} 条 (第{self.trimmed_count}次修剪)")
            return {"messages": trimmed_messages}

        return None


middleware = MessageTrimmerMiddleware(max_messages=4)  # 最多保留 4 条
agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个有帮助的助手。",
    middleware=[middleware]
    # 不使用 checkpointer
)

# 手动管理消息历史
messages = []
for i in range(6):
    print(f"\n--- 第 {i + 1} 次对话 ---")

    # 新增用户消息
    new_msg = {"role": "user", "content": f"消息{i + 1}：简短回复"}
    messages.append(new_msg)

    print(f"调用前消息数: {len(messages)}")

    # 调用 agent（middleware会修剪）
    response = agent.invoke({"messages": messages})

    # 获取完整对话（包含AI响应）
    messages = response['messages']

    print(f"调用后消息数: {len(messages)}")
    if len(messages) <= 4:
        print(f"消息列表: {[m.content[:15] for m in messages]}")
