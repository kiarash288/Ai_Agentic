# Event در دور اول حلقه:
{
    "reasoner": {                                # <--- اسم نود
        "messages": [                            # <--- خروجی این نود
            AIMessage(
                content="", 
                tool_calls=[{'name': 'power', 'args': {'a': 2, 'b': 3}, 'id': 'call_1'}]
            )
        ]
    }
}


# Event در دور دوم حلقه:
{
    "tools": {                                   # <--- اسم نود عوض شد!
        "messages": [                            # <--- خروجی این نود
            ToolMessage(
                content="8",                     # <--- جواب ریاضی
                name="power", 
                tool_call_id="call_1"
            )
        ]
    }
}


# Event در دور سوم حلقه:
{
    "reasoner": {                                # <--- دوباره نود ریزنر
        "messages": [
            AIMessage(content="جواب میشه ۸.")
        ]
    }
}



# State نهایی (ترکیب همه موارد بالا):
{
    "messages": [
        HumanMessage(content="۲ به توان ۳؟"),       # ورودی اول
        AIMessage(content="", tool_calls=[...]),    # مال دور اول (reasoner)
        ToolMessage(content="8", ...),              # مال دور دوم (tools)
        AIMessage(content="جواب میشه ۸.")           # مال دور سوم (reasoner)
    ]
}