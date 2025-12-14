import os
from dotenv import load_dotenv
from typing import TypedDict, List


from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY missing.")
    print("Please create a .env file with: GOOGLE_API_KEY=your_google_api_key_here")
    exit(1)
if not os.getenv("GROQ_API_KEY"):
    print("Error: GROQ_API_KEY missing.")
    print("Please create a .env file with: GROQ_API_KEY=your_groq_api_key_here")
    exit(1)
PDF_PATH = r"E:\AI_Engineering\Practicing_Ai_Agentic\AI\7\data\000-128 C112216.pdf"

if not os.path.exists(PDF_PATH):
    print(f"Error: File not found at {PDF_PATH}")
    print("Please create a 'data' folder and put a PDF file named 'sample.pdf' in it.")
    exit(1)

loader = PDFPlumberLoader(PDF_PATH)
docs = loader.load()
print(f"Loaded {len(docs)} pages.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks.")
# اینو بذار تو کدت ببین چی چاپ میکنه
print("--- نمونه متن استخراج شده ---")
print(splits[0].page_content[:500])  # ۵۰۰ حرف اول رو چاپ کن
print("-----------------------------")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
vector_store = FAISS.from_documents(splits, embeddings)
print("Vector Store Created Successfully! ✅")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})


class State(TypedDict):
    question: str
    context: List[str]
    answer: str


def retriever_node(state: State):
    question = state["question"]
    print(f"🔍 Searching for: {question}")
    documents = retriever.invoke(question)

    context_text = [doc.page_content for doc in documents]
    return {"context": context_text}


def generate_node(state: State):
    context = state["context"]
    question = state["question"]

    prompt = ChatPromptTemplate.from_template(
        """
    You are a helpful assistant. Answer the question based ONLY on the following context.
    If the answer is not in the context, say "I don't know based on this document."
    
    Context:
    {context}
    
    Question:
    {question}
    """
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    chain = prompt | llm
    context_str = "\n\n".join(context)
    response = chain.invoke({"context": context_str, "question": question})
    return {"answer": response.content}


builder = StateGraph(State)
builder.add_node("retrieve", retriever_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()


def main():
    while True:
        question = input("Enter your question: ")
        if question.lower() in ["exit", "quit", "bye"]:
            break
        result = graph.invoke({"question": question})
        print(f"📄 Retrieved Context (Summarized):\n{str(result['context'])[:200]}...")
        print(f"🤖 Answer:\n{result['answer']}")


if __name__ == "__main__":
    main()
