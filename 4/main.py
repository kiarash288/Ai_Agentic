import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY missing.")
    exit(1)
if not os.getenv("TAVILY_API_KEY"):
    print("Error: TAVILY_API_KEY missing. Get one at tavily.com")
    exit(1)


search_tool = TavilySearchResults(max_results=2)

tools = [search_tool]


llm = ChatGroq(
    model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.7
)
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def reasoner_node(state: State):

    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools)


builder = StateGraph(State)

builder.add_node("reasoner", reasoner_node)
builder.add_node("tools", tool_node)


builder.add_edge(START, "reasoner")


builder.add_conditional_edges(
    "reasoner",
    tools_condition,
)


builder.add_edge("tools", "reasoner")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def main():
    print("--- Web Search Agent (Powered by Tavily) ---")
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        input_message = HumanMessage(content=user_input)

        for event in graph.stream({"messages": [input_message]}, config):
            for node_name, value in event.items():
                last_msg = value["messages"][-1]

                if node_name == "reasoner":

                    if last_msg.tool_calls:
                        print(f"🌍 Searching for: {last_msg.tool_calls[0]['args']}")

                    elif last_msg.content:
                        print(f"🤖 AI: {last_msg.content}")

                elif node_name == "tools":
                    print(
                        f"✅ Search Completed. (Found {len(eval(last_msg.content))} results)"
                    )


if __name__ == "__main__":
    main()
