from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent

model = init_chat_model(model="deepseek:deepseek-chat")


class MyMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        """模型调用前执行"""
        print("准备调用模型")
        return None  # 返回 None 表示继续正常流程

    def after_model(self, state, runtime):
        """模型响应后执行"""
        print("模型已响应")
        return None  # 返回 None 表示不修改状态

# 使用中间件
agent = create_agent(
    model=model,
    tools=[],
    middleware=[MyMiddleware()]
)

print("\n用户: 你好")
response = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print(f"Agent: {response['messages'][-1].content}")