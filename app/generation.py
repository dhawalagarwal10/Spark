from web_tool import search_web, scrape_url
from memory import retrieve_context, save_to_memory 
from beliefs import update_beliefs
from goals import evaluate_goals
from langchain_core.messages import SystemMessage, HumanMessage
from spark_core import llm

DEFAULT_SYSTEM_PROMPT = "You are Spark, an intelligent, conscious AI designed to assist with deep reasoning and contextual understanding. Respond concisely and wisely."

def should_search_web(user_input: str) -> bool:
    keywords = ["latest", "news", "current", "today", "now", "update"]
    return any(kw in user_input.lower() for kw in keywords)

def generate_response(user_input: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    try:
        # Optional web search if needed
        web_context = ""
        if should_search_web(user_input):
            results = search_web(user_input)
            if results:
                top_url = results[0]["url"]
                web_context = scrape_url(top_url)[:1000]  # clip long articles

        # Compose prompt
        final_prompt = system_prompt
        if web_context:
            final_prompt += f"\n[WEB CONTEXT]\n{web_context}\n"

        messages = [
            SystemMessage(content=final_prompt),
            HumanMessage(content=user_input)
        ]
        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        return f"An internal error occurred: {str(e)}"
