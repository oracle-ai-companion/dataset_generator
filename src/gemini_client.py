import google.generativeai as genai
from src.config import Config

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)

    async def generate_response(self, prompt):
        response = await self.model.generate_content_async(prompt)
        return response.text

    async def generate_chat_response(self, messages):
        chat = self.model.start_chat(history=messages[:-1])
        response = await chat.send_message_async(messages[-1]['content'])
        return response.text