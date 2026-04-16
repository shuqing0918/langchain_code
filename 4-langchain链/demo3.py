from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

# 创建模型实例
template = PromptTemplate(
    input_variables=["role", "fruit"],
    template="{role}喜欢吃{fruit}。请简单回答，不要使用表情符号。",
)

# 创建LLM
llm = init_chat_model(model="deepseek:deepseek-chat")

# 使用新的RunnableSequence语法替代LLMChain
llm_chain = template | llm

# 输入列表
input_list = [
    {"role": "哪吒", "fruit": "水果"}, {"role": "小猪佩奇", "fruit": "苹果"}
]

# 调用LLMChain，返回结果
result = llm_chain.batch(input_list)
# 只打印内容部分，避免编码问题
for i, res in enumerate(result):
    print(f"结果 {i+1}:")
    print(res.content)
    print("=" * 50)