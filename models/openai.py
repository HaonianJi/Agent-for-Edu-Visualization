from models.base_model import BaseModel
from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class MyOpenAI(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.config.model
        self.client = OpenAI(
            api_key=self.config.api_key,  # This is the default and can be omitted
        )
        self.create_text_message = lambda text, question: {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "text", "text": question},
            ],
        }
        self.create_image_message = lambda image_path, question: {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},},
                {"type": "text", "text": question},
            ],
        }
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
        
    def predict(self, question, texts = None, images = None, history = None):
        messages = self.process_message(question, texts, images, history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        result = response.choices[0].message.content
        messages.append(self.create_ans_message(result))
        return result, messages
    
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True
    