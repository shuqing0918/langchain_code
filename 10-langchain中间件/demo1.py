from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent

model = init_chat_model(model="deepseek:deepseek-chat")


class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"[日志] 消息数: {len(state.get('messages', []))}")
        return None

    def after_model(self, state, runtime):
        last_msg = state.get('messages', [])[-1]
        print(f"[日志] 响应类型: {last_msg.__class__.__name__}")
        return None

config = {"configurable": {"thread_id": "m1"}}

agent = create_agent(
    model=model,
    tools=[],
    middleware=[LoggingMiddleware()],
    checkpointer=InMemorySaver()
)


# print("\n用户: 你好")
response = agent.invoke({"messages": [{"role": "user", "content": "你好"}]}, config=config)
# print(f"Agent: {response['messages'][-1].content}")
#
response = agent.invoke({"messages": [{"role": "user", "content": "介绍一下你自己"}]}, config=config)
# print(f"Agent: {response['messages'][-1].content}")