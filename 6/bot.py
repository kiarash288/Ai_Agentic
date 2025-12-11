import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Literal

# ایمپورت‌های مربوط به تلگرام
from langgraph import graph
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# ایمپورت‌های مربوط به هوش مصنوعی
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END


load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROK_API_KEY = os.getenv("GROQ_API_KEY")
if not TELEGRAM_TOKEN or not GROK_API_KEY:
    print('"Error: Keys are missing in .env"')


class Sentiment(BaseModel):
    mood: Literal["negative", "positive", "neutral"] = Field(
        description="تشخیص حس جمله: positive (خوشحال/مثبت)، negative (ناراحت/عصبانی)، neutral (خنی/معمولی)"
    )


class State(TypedDict):
    text: str
    sentiment: str
    final_answer: str


llm = ChatGroq(
    model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0
)
sentiment_analyzer = llm.with_structured_output(Sentiment)


def analyze_sentiment(state):
    text = state["text"]
    result = sentiment_analyzer.invoke(f"Analyze the sentiment of this text: {text}")
    mood = result.mood

    emoji = ""
    if mood == "positive":
        emoji = "😄"
    elif mood == "negative":
        emoji = "😡"
    elif mood == "neutral":
        emoji = "🙂"

    response = f"{text}\n\n Mood:{mood} {emoji}"
    return {"sentiment": mood, "final_answer": response}


builder = StateGraph(State)

builder.add_node("analyzer", analyze_sentiment)

builder.add_edge(START, "analyzer")
builder.add_edge("analyzer", END)


graph = builder.compile()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! هر چی بگی من حسش رو تشخیص میدم و تکرار میکنم. 😎"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"User said: {user_text}")

    input = {"text": user_text}
    result = graph.invoke(input)

    await update.message.reply_text(result["final_answer"])


def main():
    print("--- Telegram Bot Started ---")

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    )

    application.run_polling()


if __name__ == "__main__":
    main()
