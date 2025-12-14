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

load_dotenv()
if not os.path.exists("credentials.json"):
    print("Error: 'credentials.json' not found!")
    print("Please download it from Google Cloud Console and put it in this folder.")
    exit(1)

credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit= GmailToolkit(api_resource=api_resource)
tools=toolkit.get_tools()

llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools=llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def reasoner_node(state:State):
    return {'messages':llm_with_tools.invoke(state['messages'])}

tool_node = ToolNode(tools)

builder = StateGraph(State)

builder.add_node('reasoner', reasoner_node)
builder.add_node('tools', tool_node)

builder.add_edge(START, 'reasoner')

builder.add_conditional_edges(
    'reasoner',
    tools_condition,
)

builder.add_edge('reasoner', 'tools')

memory = MemorySaver()
graph=builder.compile(checkpointer=memory)

def main():
    print("--- Gmail Agent Started ---")
    print("Examples: 'Send an email to [email] saying hello', 'Check my latest emails'")
    
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
      
        input_message=input("You: ")
        if input_message.lower() in ['quit', 'exit']:
            break
        else:
            messages=[HumanMessage(content=input_message)]
            for event in graph.stream({"messages": messages}, config):
                for node_name, value in event.items():
                    if node_name == 'tools':
                        print("Tool Response:")
                        print(value)
                    elif node_name == 'reasoner':
                        print("Assistant:")
                        print(value)
                        print("\n")
                        
if __name__ == "__main__":
    main()