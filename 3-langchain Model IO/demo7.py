from typing import Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

model = init_chat_model(model="deepseek:deepseek-chat")


class CustomerInfo(BaseModel):
    name: str = Field(description="客户姓名")
    phone: str = Field(description="电话号码")
    email: Optional[str] = Field(None, description="邮箱")
    issue: str = Field(description="问题描述")


structured_llm = model.with_structured_output(CustomerInfo)

conversation = """
客户: 我是李明，电话 138-1234-5678，我的邮箱是123@qq.com，订单没发货
"""

info = structured_llm.invoke(f"提取客户信息：{conversation}")
print(info)
# info.name = "李明"# info.phone = "138-1234-5678"# info.issue = "订单没发货"


from typing import List


class Review(BaseModel):
    product: str
    rating: int = Field(description="评分 1-5")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")


structured_llm = model.with_structured_output(Review)

review = structured_llm.invoke("""
    iPhone 15 很棒！摄像头强大，手感好。但是价格贵，没有充电器。4分。
""")
print(review)
# review.product = "iPhone 15"
# review.rating = 4
# review.pros = ["摄像头强大", "手感好"]
# review.cons = ["价格贵", "没有充电器"]
