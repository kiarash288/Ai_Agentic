import os
from dotenv import load_dotenv
from typing import Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY missing.")
    exit(1)


class Category(BaseModel):
    label: Literal["spam", "work", "personal"] = Field(
        description="دسته‌بندی ایمیل: spam برای تبلیغات، work برای کاری، personal برای دوستانه"
    )


class State(TypedDict):
    email_content: str
    category: str
    action_log: str


llm = ChatGroq(
    model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.7
)

classifier_llm = llm.with_structured_output(Category)


def classifier_node(state: State):
    """ایمیل را می‌خواند و دسته‌بندی می‌کند"""
    content = state["email_content"]
    result = classifier_llm.invoke(f"این ایمیل را دسته‌بندی کن: {content}")

    return {"category": result.label}


def handle_spam(state: State):
    print("🗑️ Deleting Spam...")
    return {"action_log": "Moved to Trash"}


def handle_work(state: State):
    print("💼 Sending to Slack...")
    return {"action_log": "Forwarded to Manager"}


def handle_personal(state: State):
    print("💌 Saving to Archive...")
    return {"action_log": "Saved in Personal Folder"}


def route_email(state: State):

    category = state["category"]

    if category == "spam":
        return "spam_node"
    elif category == "work":
        return "work_node"
    elif category == "personal":
        return "personal_node"


builder = StateGraph(State)


builder.add_node("classifier", classifier_node)
builder.add_node("spam_node", handle_spam)
builder.add_node("work_node", handle_work)
builder.add_node("personal_node", handle_personal)


builder.add_edge(START, "classifier")


builder.add_conditional_edges(
    "classifier",
    route_email,
    {
        "spam_node": "spam_node",
        "work_node": "work_node",
        "personal_node": "personal_node",
    },
)


builder.add_edge("spam_node", END)
builder.add_edge("work_node", END)
builder.add_edge("personal_node", END)

graph = builder.compile()


def main():
    print("--- Email Classifier ---")
    while True:
        email = input("\nEmail content: ")
        if email.lower() in ["quit", "exit"]:
            break

        result = graph.invoke({"email_content": email})

        print(f"Category Detected: {result['category'].upper()}")
        print(f"Action Taken: {result['action_log']}")


if __name__ == "__main__":
    main()
