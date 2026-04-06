from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
import os

loader = PyPDFLoader("llama2.pdf")
pages = loader.load_and_split()

print(pages[:5])
print(len(pages))

# print(f"第0页：\n{pages[0]}") ## 也可通过 pages[0].page_content只获取本页内容

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./llama2.pdf")
pages = loader.load_and_split()
print(f"第1页：\n{pages[0].page_content}")
print("=" * 50)
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)

paragraphs = text_splitter.create_documents([pages[0].page_content])
for para in paragraphs:
    print(para.page_content)
    print('-------')

from langchain_openai import OpenAIEmbeddings

# embeddings_model = OpenAIEmbeddings()  ## OpenAI文本向量化模型接口的封装

from langchain_community.embeddings.dashscope import DashScopeEmbeddings

embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)

print(len(embeddings), len(embeddings[0]))
print(embeddings[0][:5])
##运行结果 (5, 1024)


embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(embedded_query[:5])

from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# db = Chroma.from_documents(paragraphs, embeddings_model) ## 一行代码搞定   

db = Chroma(collection_name="mydb", embedding_function=embeddings_model)

batch_size = 10
for i in range(0, len(paragraphs), batch_size):
    batch = paragraphs[i:i + batch_size]
    db.add_documents(batch)

query = "llama2有多少参数？"
docs = db.similarity_search(query)  ## 一行代码搞定
for doc in docs:
    print(f"{doc.page_content}\n-------\n")

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()


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
