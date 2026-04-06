# 基础 PromptTemplate
from langchain_core.prompts import PromptTemplate

# Chat Prompt 相关
from langchain_core.prompts import ChatPromptTemplate

# 单条消息 PromptTemplate
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Few-shot 提示模板
from langchain_core.prompts import FewShotPromptTemplate


# 导入LangChain中的提示模板
from langchain_core.prompts import PromptTemplate

# 创建原始模板
template = "您是一位专业的程序员。\n对于信息 {text} 进行简短描述"

# 根据原始模板创建LangChain提示模板
prompt = PromptTemplate.from_template(template)

# 打印LangChain提示模板的内容
print(prompt)
print("="*50)
print(prompt.format(text="langchain"))


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["text"],
    template="您是一位专业的程序员。\n对于信息 {text} 进行简短描述"
)
print(prompt.format(text="langchain"))


from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
# 创建模型实例
model = init_chat_model(model="deepseek:deepseek-chat")

# 输入提示
input = prompt.format(text="大模型langchain")

# 得到模型的输出
output = model.invoke(input)
# output = model.invoke("您是一位专业的程序员。对于信息 langchain 进行简短描述")

# 打印输出内容
print(output.content)