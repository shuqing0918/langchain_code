from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

# 创建模型实例
template = PromptTemplate(
    input_variables=["role", "fruit"],
    template="{role}喜欢吃{fruit}",
)

# 创建LLM
llm = init_chat_model(model="deepseek:deepseek-chat")

# 创建LLMChain
# llm_chain = prompt | llm
llm_chain = LLMChain(llm=llm, prompt=template)

# 输入列表
input_list = [
    {"role": "哪吒", "fruit": "水果"}, {"role": "小猪佩奇", "fruit": "苹果"}
]

# 调用LLMChain，返回结果
result = llm_chain.batch(input_list)
print(result)