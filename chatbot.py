from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo")
