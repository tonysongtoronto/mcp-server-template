import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()

# 2. 初始化异步客户端
client = AsyncOpenAI(
    base_url="https://api.deepseek.com", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

async def test_connection():
    print("🚀 正在检查 OpenRouter 连接...")
    
    try:
        # 发送一个简单的聊天请求
        response = await client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek 模型名称：deepseek-chat 或 deepseek-coder     
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手。"},
                {"role": "user", "content": "如果你收到了这条消息，请回复：' 连接正常'。"}
            ],
            # 顺便测试一下参数，确保能正常解析
            temperature=0.7,
            max_tokens=50
        )
        
        # 3. 解析结果
  
        
        answer = response.choices[0].message.content
        model_used = response.model  # 获取实际调用的模型名称
        request_id = response.id     # 获取 OpenRouter 的请求 ID

        print("-" * 30)
        print(f"✅ 成功连上模型！")
        print(f"实际模型: {model_used}")
        print(f"请求 ID : {request_id}")
        print(f"模型回复: {answer}")
        print("-" * 30)
        
    except Exception as e:
        print("-" * 30)
        print(f"❌ 连接发生错误！")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {e}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_connection())