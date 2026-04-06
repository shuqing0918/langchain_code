from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

# 原始字符串模板
template = "桌上有{number}个苹果，四个桃子和 3 本书，一共有几个水果?"
prompt = PromptTemplate.from_template(template)

# 创建模型实例
model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)
# 创建模型实例
llm = init_chat_model(model="deepseek:deepseek-chat")

# 创建Chain  模版 -> 大模型 -> outparse
chain = prompt | llm
# ps aux 查看所有的进程  | gerp redis
# chain = llm | prompt

# 调用Chain，返回结果
result = chain.invoke({"number": "3"})
print(result.content)