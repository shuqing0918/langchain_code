from langchain_community.retrievers import BM25Retriever


bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

print(f"\n[OK] BM25 检索器已创建")
print(f"  检索数量: k=3")

# 测试查询
results = bm25_retriever.invoke("BM25 算法")
print(results)
print(f"结果数: {len(results)}")