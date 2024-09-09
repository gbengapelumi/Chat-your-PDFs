# models.py
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv


class ModelHandler:
    def __init__(self):
        load_dotenv()
        # Initialize your models here with the specified configurations
        self.models = {
            "llama3-8b-8192": ChatGroq(model="llama3-8b-8192", temperature=0.2),
            "llama-3.1-70b-versatile": ChatGroq(
                model="llama-3.1-70b-versatile", temperature=0.2
            ),
            "llama-3.1-8b-instant": ChatGroq(
                model="llama-3.1-8b-instant", temperature=0.2
            ),
            "mixtral-8x7b-32768": ChatGroq(model="mixtral-8x7b-32768", temperature=0.2),
            "gemini-1.5-flash": ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", temperature=0.3
            ),
            "mistral-large-latest": ChatMistralAI(model="mistral-large-latest"),
        }

    '''
    def run_model(self, model_name, prompt):
        """Run a specific model based on the name."""
        model = self.models.get(model_name)
        if model:
            response = model(prompt)
            return response["choices"][0]["message"]["content"]
        else:
            return f"Model {model_name} not found."
            
            '''
