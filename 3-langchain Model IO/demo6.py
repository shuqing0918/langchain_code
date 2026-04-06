from pydantic import BaseModel, Field

class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")


# 使用 with_structured_output()
from langchain.chat_models import init_chat_model

model = init_chat_model(model="deepseek:deepseek-chat")

# 创建结构化输出的 LLM
structured_llm = model.with_structured_output(Person)

# 调用
result = structured_llm.invoke("张三是一名 30 岁的软件工程师")
print(result)

# ---

from pydantic import BaseModel, Field

class Result(BaseModel):
    title: str = Field(description="标题")
    summary: str = Field(description="简要总结")
    score: float = Field(description="评分，0-1 之间")


from langchain.chat_models import init_chat_model

model = init_chat_model(model="deepseek:deepseek-chat")

structured_llm = model.with_structured_output(Result)

res = structured_llm.invoke("总结 Transformer 的核心思想，并给出重要性评分")

print(type(res))      # <class '__main__.Result'>
print(res.title)
print(res.summary)
print(res.score)