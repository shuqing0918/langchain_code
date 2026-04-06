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
    # 你的实现
    return "晴天，温度 15°C"


from langchain.tools import tool

@tool
def add_number(a: int, b: int) -> int:
    """add two numbers."""
    return a + b

print(add_number.name)
print(add_number.description)
print(add_number.args)

res = add_number.run({"a": 10, "b": 20})
print(res)