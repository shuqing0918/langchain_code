from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent

model = init_chat_model(model="deepseek:deepseek-chat")


class CallCounterMiddleware(AgentMiddleware):
    """
    计数中间件 - 统计模型调用次数

    在中间件内部维护计数器（简单版本）
    """

    def __init__(self):
        super().__init__()
        self.count = 0  # 简单计数器

    def after_model(self, state, runtime):
        """模型响应后，增加计数"""
        self.count += 1
        print(f"\n[计数器] 模型调用次数: {self.count}")
        return None  # 不修改 state


# 需要 checkpointer 来保存自定义状态
agent = create_agent(
    model=model,
    middleware=[CallCounterMiddleware()],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "counter_test"}}

print("第一次调用:")
agent.invoke({"messages": [{"role": "user", "content": "你好"}]}, config)

print("第二次调用:")
agent.invoke({"messages": [{"role": "user", "content": "今天天气"}]}, config)
