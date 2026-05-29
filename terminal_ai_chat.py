#!/usr/bin/env python3
"""
Terminal AI Chat - 终端 AI 对话测试
简单的命令行交互式对话程序
"""

import sys
import time
from datetime import datetime


def print_banner():
    """打印欢迎信息"""
    print("=" * 60)
    print("  Terminal AI Chat - 终端 AI 对话测试")
    print("=" * 60)
    print("  命令:")
    print("    /help    - 显示帮助")
    print("    /quit    - 退出对话")
    print("    /clear   - 清屏")
    print("    /history - 显示对话历史")
    print("=" * 60)
    print()


def get_ai_response(user_input: str, history: list) -> str:
    """
    模拟 AI 响应
    实际使用时，这里应该调用真实的 AI API
    """
    # 简单的关键词响应（演示用）
    responses = {
        "hello": "你好！很高兴与你对话。",
        "hi": "你好！有什么我可以帮助你的吗？",
        "help": "我可以帮你测试终端对话功能。请直接输入你想说的话。",
        "time": f"当前时间是: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "test": "测试成功！终端对话功能正常工作。",
    }
    
    # 检查关键词
    lower_input = user_input.lower()
    for key, response in responses.items():
        if key in lower_input:
            return response
    
    # 默认响应
    default_responses = [
        "我收到了你的消息。这是一个模拟响应。",
        "理解。在实际应用中，这里会连接真实的 AI 服务。",
        "好的。终端对话测试运行正常。",
        "收到。你可以继续输入来测试对话流程。",
    ]
    
    # 根据历史长度选择不同的默认响应
    return default_responses[len(history) % len(default_responses)]


def main():
    """主函数"""
    print_banner()
    
    history = []
    
    while True:
        try:
            # 获取用户输入
            user_input = input("You: ").strip()
            
            # 空输入处理
            if not user_input:
                continue
            
            # 命令处理
            if user_input.startswith("/"):
                if user_input == "/quit" or user_input == "/q":
                    print("\nAI: 再见！感谢使用终端对话测试。")
                    break
                    
                elif user_input == "/help" or user_input == "/h":
                    print("\n命令列表:")
                    print("  /help    - 显示此帮助")
                    print("  /quit    - 退出程序")
                    print("  /clear   - 清屏")
                    print("  /history - 显示对话历史")
                    print()
                    continue
                    
                elif user_input == "/clear" or user_input == "/c":
                    print("\n" * 50)
                    print_banner()
                    continue
                    
                elif user_input == "/history" or user_input == "/hist":
                    print("\n--- 对话历史 ---")
                    for i, (user, ai) in enumerate(history, 1):
                        print(f"{i}. You: {user}")
                        print(f"   AI: {ai}")
                    print("---------------\n")
                    continue
                    
                else:
                    print(f"未知命令: {user_input}")
                    print("输入 /help 查看可用命令\n")
                    continue
            
            # 记录用户输入
            history.append((user_input, ""))
            
            # 模拟思考时间
            print("AI: ", end="", flush=True)
            time.sleep(0.3)
            
            # 获取 AI 响应
            response = get_ai_response(user_input, history)
            
            # 打印响应
            print(response)
            print()
            
            # 更新历史
            history[-1] = (user_input, response)
            
        except KeyboardInterrupt:
            print("\n\nAI: 检测到中断，正在退出...")
            break
        except EOFError:
            print("\n\nAI: 输入结束，再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")
    
    print("\n终端对话测试已结束。")
    print(f"共进行了 {len(history)} 轮对话。")


if __name__ == "__main__":
    main()
