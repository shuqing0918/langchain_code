from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)

model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("北京今天天气怎么样？")

print(response)
# 检查模型是否要求调用工具
if response.tool_calls:
    print(f"AI 决定使用工具！")
    print(f"工具调用: {response.tool_calls}")
else:
    print(f"AI 直接回答（未使用工具）")
    print(f"回复: {response.content}")