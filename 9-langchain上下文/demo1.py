from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)


agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,   # 用于生成摘要的模型
            max_tokens_before_summary=200  # 超过 200 tokens 触发摘要
        )
    ]
)

config = {"configurable": {"thread_id": "with_summary"}}

print("\n进行多轮对话...")
conversations = [
    "我叫张三，是工程师",
    "我在北京工作",
    "我喜欢编程和阅读",
    "我最近在学习 AI",
    "请总结一下我的信息"
]

for msg in conversations:
    print(f"\n用户: {msg}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config=config
    )
    # print(f"Agent: {response}")
    # print(f"Agent: {response['messages'][-1].content[:60]}...")
    print(f"Agent: {response['messages'][-1].content}")

# print(response)