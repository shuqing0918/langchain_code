from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent

model = init_chat_model(model="deepseek:deepseek-chat")


class OutputValidationMiddleware(AgentMiddleware):
    """
    输出验证中间件 - 检查响应长度

    after_model 验证输出
    """

    def __init__(self, max_length=100):
        super().__init__()
        self.max_length = max_length

    def after_model(self, state, runtime):
        """模型响应后，验证输出"""
        messages = state.get('messages', [])
        if not messages:
            return None

        last_message = messages[-1]
        content = getattr(last_message, 'content', '')

        if len(content) > self.max_length:
            print(f"\n[警告] 响应过长 ({len(content)} 字符)，已截断到 {self.max_length}")
            # 这里可以实现截断或重试逻辑
            # {}
        return None


agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个有帮助的助手。",
    middleware=[OutputValidationMiddleware(max_length=50)]
)

print("\n用户: 请详细介绍 Python 编程语言的历史、特点和应用")
response = agent.invoke({
    "messages": [{"role": "user", "content": "请详细介绍 Python 编程语言的历史、特点和应用"}]
})
print(f"Agent: {response['messages'][-1].content[:50]}...")
