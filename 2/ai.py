import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

load_dotenv()
if not os.getenv('GROQ_API_KEY'):
    print("Error: GROQ_API_KEY not found in .env file")

class Joke(BaseModel):
    setup:str=Field(description='مقدمه یا بخش اول جوک که شنونده را آماده می‌کند')
    punchline:str=Field(description='بخش نهایی و خنده‌دار جوک')
    rating:int=Field(description='میزان بامزگی جوک از ۱ تا ۱۰ به نظر خودت',ge=1,le=10)
class State(TypedDict):
    topic:str
    generated_joke:Joke
llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)
structured_llm=llm.with_structured_output(Joke)
def generate_joke_node(state:State):
    topic=state['topic']
    prompt=f"یک جوک خیلی خنده‌دار درباره '{topic}' بگو."
    response=structured_llm.invoke(prompt)
    return{'generated_joke':response}

builder=StateGraph(State)
builder.add_node('joke_generator',generate_joke_node)
builder.add_edge(START,'joke_generator')
builder.add_edge('joke_generator',END)
graph=builder.compile()
def main():
    print('--- Structured Joke Generator ---')
    while True:
                user_topic=input('\nموضوع جوک رو بگو (یا quit): ')
                if user_topic.lower() in ["quit", "exit"]:
                    break
                result=graph.invoke('topic':user_topic)
                joke=result['generated_joke']
                print(f'setup: {joke.setup}')
                print(f'punchline:{joke.punchline}')
                print(f'rating:{joke.rating}/10')
if __name__ == "__main__":
    main()
print(m)
def tab():
                
