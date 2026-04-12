from dotenv import load_dotenv
load_dotenv()  # .env 中应包含 QIANWEN_API_KEY 和 DEEPSEEK_API_KEY

import os
from langchain_openai import ChatOpenAI

# 通义千问模型
qwllm = ChatOpenAI(
    api_key=os.getenv("QIANWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus"
)

# DeepSeek 模型（如果需要可保留）
dpllm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

# 测试普通调用
response = qwllm.invoke("你是谁？")
print(response.content)
print("=" * 50)

# 测试流式输出
for chunk in qwllm.stream("写一段诗歌"):
    print(chunk.content, end="", flush=True)
print()  # 换行