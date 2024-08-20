import google.generativeai as genai
from dotenv import load_dotenv
import os

class GeminiClient:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.models = {
            "gemini-pro": genai.GenerativeModel("gemini-pro"),
            "gemini-1.5-pro": genai.GenerativeModel("gemini-1.5-pro"),
            "gemini-1.5-flash": genai.GenerativeModel("gemini-1.5-flash")
        }

    async def generate_response(self, prompt, model):
        if model not in self.models:
            raise ValueError(f"Invalid model name: {model}")
        response = await self.models[model].generate_content_async(prompt)
        return response.text

    async def generate_chat_response(self, messages, model):
        if model not in self.models:
            raise ValueError(f"Invalid model name: {model}")
        chat = self.models[model].start_chat(history=messages[:-1])
        response = await chat.send_message_async(messages[-1]['content'])
        return response.text