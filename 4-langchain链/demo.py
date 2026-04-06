# 导入LangChain中的提示模板
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os


# 原始字符串模板
template = "桌上有{number}个苹果，四个桃子和 3 本书，一共有几个水果?"

# 创建LangChain模板
prompt_temp = PromptTemplate.from_template(template)

# 根据模板创建提示
prompt = prompt_temp.format(number=2)

from langchain.chat_models import init_chat_model

# 创建模型实例
model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)
# 传入提示，调用模型返回结果
result = model.invoke(prompt)
print(result.content)

# ---

from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

# 原始字符串模板
template = "桌上有{number}个苹果，四个桃子和 3 本书，一共有几个水果?"

# 创建模型实例
llm = init_chat_model(model="deepseek:deepseek-chat")

# 创建LLMChain
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(template)
)

# 调用LLMChain，返回结果
result = llm_chain.invoke({"number":2})
print(result)