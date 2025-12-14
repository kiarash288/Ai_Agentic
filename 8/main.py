import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY missing.")

llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

class State(TypedDict):
    topic:str
    draft:str
    critique:str
    revision_number:int

def writer_node(state:State):
    topic=state['topic']
    draft=state.get('draft')
    critique=state.get('critique')
    revision_number=state.get('revision_number',0) + 1

    if critique:
        prompt=f"""
        You are a professional email writer.
        Original Topic: {topic}
        Current Draft: {draft}
        
        Critique to address: {critique}
        
        Please write a NEW, improved version of the email that addresses the critique.
        Return ONLY the email text.
        """
        chain=prompt | llm
        response=chain.invoke({'topic':topic, 'draft':draft, 'critique':critique})
        return{'draft':response.content, 'revision_number':revision_number}
    else:
        prompt=f"""
        Write a professional email about: {topic}.
        Return ONLY the email text.
        """
        print(f"✍️ Revising draft (Version {revision_number})...")
        chain=prompt | llm
        response=chain.invoke({'topic':topic})
        return{'draft':response.content, 'revision_number':revision_number}

def critic_node(state:State):
    draft=state['draft']

    prompt=f'''
        You are a strict editor. Review the following email draft.
    Critique it for:
    1. Tone (should be professional but warm)
    2. Clarity
    3. Conciseness
    
    Draft:
    {draft}
    
    Provide a short paragraph of constructive criticism/feedback.
    '''
    print("🧐 Critiquing draft...")
    chain=prompt | llm
    response=chain.invoke({'draft':draft})
    return{'critique':response.content}

def should_continue(state:State):
    revision_number=state.get('revision_number',0)
    if revision_number > 2:
        return END
    else:
        return 'critic'


builder=StateGraph(State)
builder.add_node('writer', writer_node)
builder.add_node('critic', critic_node)

builder.add_edge(START, 'writer')
builder.add_conditional_edges(
    'writer',
    should_continue,
    {
        'critict':'critic',
        END:END
    }
)

builder.add_edge('critic', 'writer')
graph=builder.compile()
def main():
    print("--- Reflection Agent (Email Writer) ---")
    while True:
        topic=input("Email Topic (e.g., Request for salary raise): ")
        initial_state={
            'topic':topic,
            'revision_number':0,
            'draft':'',
            'critique':''
        }
        result=graph.invoke(initial_state)
        print("\n🎉 Final Email Draft:")
        print(result['draft'])
        