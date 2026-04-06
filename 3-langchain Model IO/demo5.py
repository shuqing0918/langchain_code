from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


# 初始化语言模型
model = init_chat_model(model="deepseek:deepseek-chat")

# 创建解析器
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

output_parser = StrOutputParser()
# output_parser = JsonOutputParser()

# 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的程序员"),
    ("user", "{input}")
])


# 将提示和模型合并以进行调用
chain = prompt | model | output_parser
# chain = prompt | model

# res = chain.invoke({"input": "langchain是什么? 问题用q 回答用a 返回一个JSON格式"})
res = chain.invoke({"input": "大模型中的langchain是什么?"})
print(res)

# -----

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# 初始化语言模型
model = init_chat_model(model="deepseek:deepseek-chat")

# 创建解析器
from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()

# 提示模板
template = "生成5个列表{text} 用中文回答.\n\n{format_instructions}"

# 根据提示模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template, partial_variables={"format_instructions": output_parser.get_format_instructions()},)
print(output_parser.get_format_instructions())

# 提示模板与输出解析器传递输出
# chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())

# 将提示和模型合并以进行调用
chain = prompt | model | output_parser
res = chain.invoke({"text": "颜色"})
print(res)