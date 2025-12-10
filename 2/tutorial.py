from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="تو یک کمدین بی‌ادب و بامزه هستی."),
    HumanMessage(content=prompt)
    
]

# اینجا ورودی یک لیست است
response = structured_llm.invoke(messages)

{
    "topic": "cat",  # همان قبلی باقی مانده
    
    # این قسمت جدید اضافه شده:
    "generated_joke": Joke(
        setup="چرا گربه روی کامپیوتر نشست؟",
        punchline="چون می‌خواست حواسش به ماوس باشه!",
        rating=8
    )
}