import requests
import os
from dotenv import load_dotenv

load_dotenv()
from langchain.tools import tool


# 获取天气信息的函数
@tool
def get_weather(city):
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"

    返回:
        天气JSON信息
    """
    apiUrl = 'http://apis.juhe.cn/simpleWeather/query'  # 接口请求URL
    apiKey = os.getenv("WEATHER_API_KEY")  # 在个人中心->我的数据,接口名称上方查看
    # print(apiKey)
    # 接口请求入参配置
    requestParams = {
        'key': apiKey,
        'city': city,
    }

    # 发起接口网络请求
    response = requests.get(apiUrl, params=requestParams)
    # print(response)
    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        return responseResult
        # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
        # print(responseResult) 
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')

# result = get_weather("长沙")
# print(result)


from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个有帮助的助手，可以查询天气信息。"  # 可选
)

# response = agent.invoke({
#     "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
# })
# for message in response["messages"]:
#     message.pretty_print()

response = agent.invoke({
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}]
})

for message in response["messages"]:
    message.pretty_print()