import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. تنظیمات اولیه
load_dotenv()

# بررسی وجود فایل کردینشال گوگل
if not os.path.exists("credentials.json"):
    print("Error: 'credentials.json' not found!")
    print("Please download it from Google Cloud Console and put it in this folder.")
    exit(1)

# 2. راه‌اندازی ابزارهای Gmail
# این بخش به طور خودکار مرورگر را باز می‌کند تا شما لاگین کنید
credentials = get_gmail_credentials(
    token_file="token.json",  # توکن اینجا ذخیره می‌شود تا دفعات بعد لاگین نخواهد
    scopes=["https://mail.google.com/"], # دسترسی کامل
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

# استخراج ابزارها از تولکیت
tools = toolkit.get_tools()
print(f"✅ Gmail Tools Loaded: {[t.name for t in tools]}")

# 3. تعریف مدل و اتصال ابزارها
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 4. تعریف استیت
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 5. تعریف نودها
def reasoner_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

# 6. ساخت گراف
builder = StateGraph(State)

builder.add_node("reasoner", reasoner_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 7. اجرا
def main():
    print("--- Gmail Agent Started ---")
    print("Examples: 'Send an email to [email] saying hello', 'Check my latest emails'")
    
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        # اضافه کردن یک دستور سیستمی برای رفتار بهتر
        # به او می‌گوییم تو دستیار ایمیل هستی.
        msgs = [HumanMessage(content=user_input)]
        
        for event in graph.stream({"messages": msgs}, config):
            for node_name, value in event.items():
                last_msg = value["messages"][-1]
                
                if node_name == "reasoner":
                    if last_msg.tool_calls:
                        print(f"🛠️ AI is calling tool: {last_msg.tool_calls[0]['name']}")
                    elif last_msg.content:
                        print(f"🤖 AI: {last_msg.content}")
                
                elif node_name == "tools":
                    print("✅ Tool Executed.")

if __name__ == "__main__":
    main()