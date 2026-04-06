from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./langchain.txt", encoding="utf-8")
pages = loader.load()
# print(pages[:3])
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)

chunks = text_splitter.split_documents(pages)
# print(chunks)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

results = vector_retriever.invoke("BM25 算法")
print(results)