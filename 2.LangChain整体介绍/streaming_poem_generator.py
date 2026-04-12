from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import sys

# 加载环境变量
load_dotenv()

# 使用DeepSeek模型
model = ChatDeepSeek(model="deepseek-chat")

# 测试流式输出
if __name__ == "__main__":
    print("流式输出 - 诗歌生成器")
    print("=" * 50)
    
    # 用户提示词
    user_input = "帮我写一首诗"
    print(f"用户输入: {user_input}")
    print("\n生成的诗歌:")
    print("-" * 50)
    
    # 构建诗歌生成提示
    poem_prompt = "请创作一首优美的诗歌，要有意境，语言流畅，富有情感。"
    
    # 直接使用模型的流式输出
    from langchain_core.messages import HumanMessage
    
    # 流式生成诗歌
    print("正在生成诗歌...\n")
    for chunk in model.stream([HumanMessage(content=poem_prompt)]):
        if hasattr(chunk, "content") and chunk.content:
            # 流式输出内容
            sys.stdout.write(chunk.content)
            sys.stdout.flush()
    
    print("\n" + "-" * 50)
    print("诗歌生成完成！")
