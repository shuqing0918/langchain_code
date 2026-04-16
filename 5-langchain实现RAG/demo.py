from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.embeddings import DashScopeEmbeddings
# 使用千问向量模型
embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("QIANWEN_API_KEY")
)
# from langchain_community.embeddings import HuggingFaceEmbeddings

# embeddings_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

#把字符串转成向量
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./llama2.pdf")
pages = loader.load_and_split()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)

paragraphs = text_splitter.create_documents([pages[0].page_content])

from langchain_chroma import Chroma

# db = Chroma.from_documents(paragraphs, embeddings_model) ## 一行代码搞定   

db = Chroma(collection_name="mydb", embedding_function=embeddings_model)

batch_size = 10
for i in range(0, len(paragraphs), batch_size):
    batch = paragraphs[i:i + batch_size]
    db.add_documents(batch)

query = "llama2有多少参数？"
# docs = db.similarity_search(query)  ## 一行代码搞定

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

# 将 vectorstore 封装为工具
@tool
def search_kb(query: str) -> str:
    """搜索知识库"""
    docs = db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)

# 创建 RAG Agent
agent = create_agent(
    model=model,
    tools=[search_kb],
    system_prompt="""你是助手。使用search_kb工具检索信息，然后回答问题。"""
)

# 问答
response = agent.invoke({
    "messages": [{"role": "user", "content": "llama2有多少参数？"}]
})

for message in response["messages"]:
    message.pretty_print()
