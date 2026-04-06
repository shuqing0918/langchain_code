from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent

model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"

    返回:
        天气信息字符串
    """
    # 模拟天气数据（实际应用中应调用真实API）
    weather_data = {
        "北京": "晴天，温度 15°C，空气质量良好",
        "上海": "多云，温度 18°C，有轻微雾霾",
        "深圳": "阴天，温度 22°C，可能有小雨",
        "成都": "小雨，温度 12°C，湿度较高"
    }

    return weather_data.get(city, f"抱歉，暂时没有{city}的天气数据")


agent = create_agent(
    model=model,
    tools=[get_weather],  # 只给一个工具
    system_prompt="你是一个有帮助的助手，可以查询天气信息。"
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
})

for message in response["messages"]:
    message.pretty_print()