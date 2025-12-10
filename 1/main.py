import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("Error: OPENAI_API_KEY not found in .env file")


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatGroq(
    model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0
)


def chatbot_node(state: State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def main():
    print("--- Chatbot Started (Type 'quit' to exit) ---")
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("User:")
        if user_input.lower() in ["quit", "exit"]:
            break
        input_message = HumanMessage(content=user_input)
        for event in graph.stream({"messages": [input_message]}, config):
            for value in event.values():
                print(f"Assistant:{value['messages'][-1].content}")


if __name__ == "__main__":
    main()
