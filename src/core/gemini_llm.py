from src.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash",api_key=settings.google_api_key)
