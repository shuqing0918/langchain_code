# 导入和使用 WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()
import bs4
import os


loader = WebBaseLoader(
    web_path="https://www.gov.cn/xinwen/2020-06/01/content_5516649.htm",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="UCAP-CONTENT"))
)
docs = loader.load()
# print(docs)

# 对于嵌入模型，这里通过 API调用
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# qwen的embding模型
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

#使用此嵌入模型将文档摄取到矢量存储中
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 使用分割器分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
print(len(documents))
# 向量存储  embeddings 会将 documents 中的每个文本片段转换为向量，并将这些向量存储在 FAISS 向量数据库中
vector = FAISS.from_documents(documents, embeddings)