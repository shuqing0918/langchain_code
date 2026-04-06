from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)


db_path = "multi_user.sqlite"
with SqliteSaver.from_conn_string(db_path) as checkpointer:
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=checkpointer
    )

    # 用户 A
    print("\n[用户 A 的对话]")
    config_a = {"configurable": {"thread_id": "user_alice"}}
    agent.invoke(
        {"messages": [{"role": "user", "content": "我是 Alice，我喜欢编程"}]},
        config_a
    )
    print("Alice: 我是 Alice，我喜欢编程")

    # 用户 B
    print("\n[用户 B 的对话]")
    config_b = {"configurable": {"thread_id": "user_bob"}}
    agent.invoke(
        {"messages": [{"role": "user", "content": "我是 Bob，我喜欢设计"}]},
        config_b
    )
    print("Bob: 我是 Bob，我喜欢设计")

    # 回到用户 A
    print("\n[用户 A 继续对话]")
    response_a = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么？"}]},
        config_a
    )
    print(f"Alice: 我喜欢什么？")
    print(f"Agent: {response_a['messages'][-1].content}")

    # 回到用户 B
    print("\n[用户 B 继续对话]")
    response_b = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么？"}]},
        config_b
    )
    print(f"Bob: 我喜欢什么？")
    print(f"Agent: {response_b['messages'][-1].content}")