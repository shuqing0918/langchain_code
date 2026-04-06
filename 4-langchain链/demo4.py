import os
from dotenv import load_dotenv

# 基础依赖
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# 向量库
from langchain_community.vectorstores import FAISS

# Retrieval + Stuff Chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# 环境变量
load_dotenv()


# Step 1: 准备文档
docs = [
    Document(page_content="LangChain 是一个用于构建大语言模型应用的框架。"),
    Document(page_content="LangChain 支持 RAG、Agent、Tool Calling 等能力。"),
    Document(page_content="RAG 的核心思想是：检索外部知识 + 大模型生成。"),
]

# Step 2: 构建向量库
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Step 3: 定义 Prompt（⚠️ 必须有 {context}）
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个严谨的 AI 助手，请仅基于以下文档内容回答问题。\n\n"
     "文档内容：{context}"
    ),
    ("human", "{input}")
])


# Step 4: 初始化 LLM
llm = init_chat_model(model="deepseek:deepseek-chat")


# Step 5: 创建 Stuff Documents Chain
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# Step 6: 创建 Retrieval Chain（RAG）
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)


# Step 7: 调用
result = rag_chain.invoke({
    "input": "什么是 RAG？LangChain 在其中起什么作用？"
})

print("====== 最终回答 ======")
print(result["answer"])