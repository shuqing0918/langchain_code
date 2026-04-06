from dotenv import load_dotenv
load_dotenv()  # .env DASHSCOPE_API_KEY="xxx"
import os
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek


# llm = ChatOpenAI()
qwllm = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   model="qwen-plus")

dpllm = ChatOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                   base_url="https://api.deepseek.com",
                   model="deepseek-chat")

# 直接提供问题，并调用llm
response = dpllm.invoke("什么是大模型？")
print(response)
print("=" * 50)
print(response.content)


from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)

# 流式输出
# for chunk in model.stream("写一段诗歌"):
#     print(chunk.content, end="", flush=True)

# print(model.invoke("写一段诗歌").content)



