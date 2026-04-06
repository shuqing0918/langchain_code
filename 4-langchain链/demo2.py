from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_experimental.utilities import PythonREPL
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

template = """Write some python code to solve the user's problem.

Return only python code in Markdown format, e.g.:

```python
....
```"""
prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

model = init_chat_model(model="deepseek:deepseek-chat")


def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]


# PythonREPL().run 就是调用了一下 exec 函数执行代码
# chain = prompt | model | StrOutputParser()
chain = prompt | model | StrOutputParser() | _sanitize_output | PythonREPL().run
# chain = prompt | model | StrOutputParser() | _sanitize_output
result = chain.invoke({"input": "[1,4,2]给这个列表写一个升序的排序方法"})

print(result)
