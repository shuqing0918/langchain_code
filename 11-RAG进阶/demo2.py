from langchain_classic.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[1, 0]  # 稍微偏向向量搜索
)

print(f"\n[OK] 混合检索器已创建")
print(f"  组合: BM25 (40%) + Vector (60%)")
print(f"  算法: RRF (Reciprocal Rank Fusion)")


# 对比测试
test_queries = [
    ("语义查询", "LangChain 的主要功能是什么？"),
    ("精确匹配", "langchain>=1.0.0"),
    ("混合查询", "BM25 算法如何工作？"),
]

print(f"\n对比测试:")
for query_type, query in test_queries:
    print(f"\n  [{query_type}] {query}")

    # BM25 结果
    bm25_results = bm25_retriever.invoke(query)
    bm25_preview = bm25_results[0].page_content[:90].replace("\n", " ") if bm25_results else "无"

    # 向量结果
    vector_results = vector_retriever.invoke(query)
    vector_preview = vector_results[0].page_content[:90].replace("\n", " ") if vector_results else "无"

    # 混合结果
    ensemble_results = ensemble_retriever.invoke(query)
    ensemble_preview = ensemble_results[0].page_content[:90].replace("\n", " ") if ensemble_results else "无"

    print(f"BM25: {bm25_preview}...")
    print(f"Vector: {vector_preview}...")
    print(f"Hybrid: {ensemble_preview}...")