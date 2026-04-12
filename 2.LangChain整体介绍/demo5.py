import os
from dotenv import load_dotenv
import bs4

# 1. 加载环境变量
load_dotenv()

# 2. 向量库准备
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path="https://www.runoob.com/ml/ml-intro.html",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="col middle middle-column"))
)
## 演示加载网页文档内容
docs = loader.load()
# 文档集合
docs = [
    Document(page_content="北京是中国的首都，也是政治文化中心。故宫博物馆也在北京"),
    Document(page_content="上海是中国的经济中心，金融业发达。"),
    Document(page_content="深圳是中国的科技创新城市。"),
    Document(page_content="杭州以互联网产业著称，是阿里巴巴总部所在地。"),
]

# Embedding + FAISS
from langchain_community.embeddings import DashScopeEmbeddings

# 使用千问向量模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("QIANWEN_API_KEY")
)
# 从文档集合创建向量库
vectorstore = FAISS.from_documents(docs, embeddings)

# 3. 定义 RAG 检索工具
from langchain.tools import tool

@tool
def rag_search(query: str) -> str:
    """利用向量库检索并返回最相关的文档内容"""
    results = vectorstore.similarity_search(query, k=1)
    # 把每个文档内容组合成一个字符串
    return "\n\n".join([doc.page_content for doc in results])


# 4. 初始化 LLM（DeepSeek Chat）
from langchain.chat_models import init_chat_model

# llm = init_chat_model(
#     # model="deepseek:deepseek-chat"
#     model="qwen-plus"
# )
from langchain_openai import ChatOpenAI
# 也可以初始化Qwen模型
llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=os.getenv("QIANWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 5. 创建 RAG Agent
from langchain.agents import create_agent

system_prompt = """你是一个检索增强问答助手 (RAG)。针对用户的问题，
如果需要背景知识，请调用 rag_search 工具获取相关文档片段并基于此回答。"""

agent = create_agent(
    model=llm,
    tools=[rag_search],
    system_prompt=system_prompt
)

# 6. 使用 RAG Agent 回答问题
if __name__ == "__main__":
    # 构造输入消息
    user_query = "介绍北京"
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": user_query}
        ]
    })
    for message in response["messages"]:
        message.pretty_print()