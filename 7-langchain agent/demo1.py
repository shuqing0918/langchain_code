from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# 查询 Tavily 搜索 API 并返回 json 的工具
search = TavilySearchResults()


@tool
def web_search(query: str) -> str:
    """
    使用 Tavily 搜索互联网信息

    参数:
        query: 搜索的关键词，如"今日金价"

    返回:
        搜索结果字符串
    """
    results = search.invoke(query)
    # print(results)
    return "\n".join(f"{r['title']}: {r['content']}" for r in results)


# web_search.invoke("今日国内金价是多少？")

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


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)
agent = create_agent(
    model=model,
    tools=[get_weather, web_search],
    system_prompt="你是一个有用的助手。"
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "上海的天气怎么样？"}]
})

# response = agent.invoke({
#     "messages": [{"role": "user", "content": "今日国内金价是多少？"}]
# })

response = agent.invoke({
    "messages": [{"role": "user", "content": "在上海的天气基础上加3度是多少？"}]
})

for message in response["messages"]:
    message.pretty_print()
