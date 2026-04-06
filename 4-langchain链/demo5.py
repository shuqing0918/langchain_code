from langchain_community.utilities import SQLDatabase

# 连接 sqlite 数据库
# db = SQLDatabase.from_uri("sqlite:///demo.db") 

# 连接 MySQL 数据库
db_user = "root"
db_password = "root"
db_host = "127.0.0.1"
db_port = "3306"
db_name = "func"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

print("那种数据库：", db.dialect)
print("获取数据表：", db.get_usable_table_names())
# 执行查询
res = db.run("SELECT count(*) FROM students;")
print("查询结果：", res)

from langchain.chat_models import init_chat_model
from langchain_classic.chains import create_sql_query_chain

# 初始化大模型
llm = init_chat_model(model="deepseek:deepseek-chat")

chain = create_sql_query_chain(llm=llm, db=db)
# user = input("输入要查询的需求:")

# response = chain.invoke({"question": "数据表orders哪个用户消费最高？"})
# response = chain.invoke({"question": "查询商品分类表中的商品分类有多少种？只返回可以执行的SQL语句"})
# response = chain.invoke({"question": "查询一班的学生数学成绩是多少？只返回可以执行的SQL语句"})
# response = chain.invoke({"question": user})
# 限制使用的表
response = chain.invoke({"question": "查询一班的学生数学成绩是多少？只返回可以执行的SQL语句", "table_names_to_use": ["students"]})
print(response)


def _sanitize_output(text: str):
    _, after = text.split("```sql")
    return after.split("```")[0]


result = _sanitize_output(response)
res = db.run(result)
print("查询结果：", res)
