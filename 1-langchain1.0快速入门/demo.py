from langchain.agents import create_agent
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

# https://www.tavily.com/   .env TAVILY_API_KEY
web_search = TavilySearchResults(max_results=2)
model = ChatDeepSeek(model="deepseek-chat")

# 定义查询订单状态的函数
# def query_order_status(order_id):
#     """查询订单状态，根据订单号返回订单的发货和送达信息。"""
#     if order_id == "1024":
#         return "订单 1024 的状态是：已发货，预计送达时间是 3-5 个工作日。"
#     else:
#         return f"未找到订单 {order_id} 的信息，请检查订单号是否正确。"


@tool("query_order_status", description="根据订单号查询订单状态")
def query_order_status(order_id: str) -> str:
    if order_id == "1024":
        return "订单 1024 的状态是：已发货，预计送达时间是 3-5 个工作日。"
    else:
        return f"未找到订单 {order_id} 的信息，请检查订单号是否正确。"


# 定义退款政策说明函数
def refund_policy(keyword):
    """查询退款政策，返回退款规则说明。"""
    print("keyword = ", keyword)
    return "我们的退款政策是：在购买后7天内可以申请全额退款，需提供购买凭证。"



agent = create_agent(
    # model="openai:gpt-4o",  # 支持字符串标识或模型实例[citation:3]
    model=model,  # 支持字符串标识或模型实例[citation:3]
    tools=[query_order_status, refund_policy, web_search],  # 赋予智能体工具调用能力
    system_prompt="你是一个专业的客服助手，帮助用户查询订单信息还可以调用工具帮助用户解决问题"  # 定义角色和行为[citation:3]
)

# 调用智能体
result = agent.invoke({
    # "messages": [{"role": "user", "content": "请帮我查询2024年诺贝尔物理学奖得主是谁？?"}]
    "messages": [{"role": "user", "content": "2026年的美国总统是谁？"}]
})


print(result)
print(result['messages'][-1].content)