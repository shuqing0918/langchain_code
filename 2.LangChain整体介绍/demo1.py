from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


model = init_chat_model(
    model="deepseek:deepseek-chat",
    temperature=0.1,
    max_tokens=2000
)

# system role
# user role
from langchain_core.messages import SystemMessage, HumanMessage


messages = [
    SystemMessage(content="你是于老师的个人助理。你叫小沐"),
    HumanMessage(content="我叫同学小张"),
    # AIMessage(content="好的老板，你有什么吩咐？"),
    # HumanMessage(content="我是谁？")
    HumanMessage(content="你是谁？")
]

# messages = [
#     {"role": "system", "content": "你是一个专业的数学老师。"},
#     {"role": "user", "content": "什么是斐波那契数列？"},
# ]

response = model.invoke(messages)
print(response.content)