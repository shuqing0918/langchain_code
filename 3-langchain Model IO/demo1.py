from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

# system_template = "你是一个数学家，你可以计算任何算式"
system_template = "你是一个翻译专家,擅长将 {input_language} 语言翻译成 {output_language}语言."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template),
])
print(chat_prompt)

# 创建模型实例
model = init_chat_model(model="deepseek:deepseek-chat")

messages = chat_prompt.format_messages(input_language="中文", output_language="英文", text="明天星期 又要上班咯")
print(messages)
# 得到模型的输出
output = model.invoke(messages)
# 打印输出内容
print(output.content)