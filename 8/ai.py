import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 1. تنظیمات اولیه
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY missing.")
    exit(1)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. تعریف State
class State(TypedDict):
    topic: str          # موضوع ایمیل
    draft: str          # متن پیش‌نویس فعلی
    critique: str       # نقدی که روی متن شده
    revision_number: int # شمارنده تعداد دفعات اصلاح

# 3. تعریف نودها (Nodes)

def writer_node(state: State):
    """
    این نود دو نقش دارد:
    ۱. اگر بار اول باشد: پیش‌نویس می‌نویسد.
    ۲. اگر نقد وجود داشته باشد: متن را اصلاح می‌کند.
    """
    topic = state["topic"]
    draft = state.get("draft")
    critique = state.get("critique")
    revision_number = state.get("revision_number", 0) + 1 # شماره نسخه را یکی زیاد کن

    # اگر نقد وجود دارد، یعنی باید اصلاح کنیم
    if critique:
        prompt = f"""
        You are a professional email writer.
        Original Topic: {topic}
        Current Draft: {draft}
        
        Critique to address: {critique}
        
        Please write a NEW, improved version of the email that addresses the critique.
        Return ONLY the email text.
        """
        print(f"✍️ Revising draft (Version {revision_number})...")
    
    # اگر نقد نیست، یعنی بار اول است
    else:
        prompt = f"""
        Write a professional email about: {topic}.
        Return ONLY the email text.
        """
        print("📝 Writing initial draft...")

    response = llm.invoke(prompt)
    
    return {
        "draft": response.content, 
        "revision_number": revision_number
    }

def critic_node(state: State):
    """
    این نود نقش یک مدیر سخت‌گیر را بازی می‌کند و ایرادات را می‌گوید.
    """
    draft = state["draft"]
    
    prompt = f"""
    You are a strict editor. Review the following email draft.
    Critique it for:
    1. Tone (should be professional but warm)
    2. Clarity
    3. Conciseness
    
    Draft:
    {draft}
    
    Provide a short paragraph of constructive criticism/feedback.
    """
    
    print("🧐 Critiquing draft...")
    response = llm.invoke(prompt)
    return {"critique": response.content}

# 4. شرط توقف حلقه (Logic)
def should_continue(state: State):
    # اگر ۳ بار اصلاح کردیم، کافیه. برو بیرون.
    if state["revision_number"] > 2:
        return END
    # در غیر این صورت، برو نقد کن
    return "critic"

# 5. ساخت گراف
builder = StateGraph(State)

builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)

builder.add_edge(START, "writer")

# بعد از نویسنده، تصمیم می‌گیریم (نقد کنیم یا تمام؟)
builder.add_conditional_edges(
    "writer",
    should_continue,
    {
        "critic": "critic", # اگر گفت critic، برو نقد کن
        END: END            # اگر گفت END، تمام کن
    }
)

# بعد از نقد، حتماً باید برگردیم به نویسنده تا اصلاح کند (حلقه)
builder.add_edge("critic", "writer")

graph = builder.compile()

# 6. اجرا
def main():
    print("--- Reflection Agent (Email Writer) ---")
    topic = input("Email Topic (e.g., Request for salary raise): ")
    
    # وضعیت اولیه (شمارنده صفر)
    initial_state = {
        "topic": topic,
        "revision_number": 0,
        "draft": "",
        "critique": ""
    }
    
    # اینجا برای اینکه خروجی نهایی رو بگیریم از invoke استفاده می‌کنیم
    # اما چون گراف حلقه داره، ممکنه بخوایم مراحل رو ببینیم (که پرینت کردیم)
    result = graph.invoke(initial_state)
    
    print("\n" + "="*40)
    print("🚀 FINAL EMAIL:")
    print("="*40)
    print(result["draft"])
    
    print("\n" + "="*40)
    print(f"Total Revisions: {result['revision_number']}")

if __name__ == "__main__":
    main()