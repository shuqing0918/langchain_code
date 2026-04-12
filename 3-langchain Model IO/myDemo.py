from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# 修复变量名拼写
prompt = PromptTemplate(
    input_variables=["text"],
    template="您是一位专业的程序员。\n对于信息 {text} 进行简短描述"
)

java_prompt = prompt.format(text="java")

llm = init_chat_model(
    model="deepseek:deepseek-chat"
)

# 创建智能体（需要提供tools参数）
agent = create_agent(
    model=llm,
    tools=[],  # 这里可以添加工具
    system_prompt=java_prompt,
)

# 调用智能体并获取响应
resp = agent.invoke({
    "messages": [{"role": "user", "content": "请介绍Java的主要特点"}]
})

# 打印响应内容
print(resp["messages"][-1].content)