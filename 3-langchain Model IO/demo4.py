from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
import os

# 初始化 OpenAIEmbeddings 实例
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Qwen
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
# 定义一个文本字符串
text = "大模型"

# 嵌入文档
doc_result = embeddings.embed_documents([text])
# [ [] ]
print(doc_result[0][:5])

# 嵌入查询
query_result = embeddings.embed_query(text)
print(query_result[:5])