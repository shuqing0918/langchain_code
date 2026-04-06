from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

examples = [
    {"input": "2+2", "output": "4", "description": "加法运算"},
    {"input": "5-2", "output": "3", "description": "减法运算"},
]

from langchain_core.prompts import PromptTemplate

# 创建提示模板，配置一个提示模板，将一个示例格式化为字符串
prompt_template = "你是一个数学专家,算式： {input} 值： {output} 使用： {description} "

# 这是一个提示模板，用于设置每个示例的格式
prompt_sample = PromptTemplate.from_template(prompt_template)

# input="2+2", output="4", description="加法运算"
# print(prompt_sample.format_prompt(**examples[1]))
print(prompt_sample.format_prompt(input="2+2", output="4", description="加法运算"))


# 创建一个FewShotPromptTemplate对象
from langchain_core.prompts import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt_sample,
    suffix="你是一个数学专家,算式: {input}  值: {output}",
    input_variables=["input", "output"]
)
print(prompt.format(input="2*5", output="10"))  # 你是一个数学专家,算式: 2*5  值: 10

model = init_chat_model(model="deepseek:deepseek-chat")

result = model.invoke(prompt.format(input="2*5", output="10"))
print(result.content)  # 使用: 乘法运算