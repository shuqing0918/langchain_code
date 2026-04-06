from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()


# 创建检索工具
@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索相关信息（混合检索）"""
    docs = ensemble_retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs[:2]])  # 只取前2个


print(f"[OK] 工具已创建: search_knowledge_base")

model = init_chat_model(model="deepseek:deepseek-chat")

# 创建Agent
agent = create_agent(
    model=model,
    tools=[search_knowledge_base],
    system_prompt="""
        你是一个有用的助手。
        使用 search_knowledge_base 工具搜索相关信息，然后回答问题。

        注意：
        1. 优先使用检索到的信息
        2. 如果信息不足，诚实告知
        3. 回答要简洁准确
    """
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "LangChain 有哪些核心组件?"}]
})

print(f"回答: {response['messages'][-1].content}")