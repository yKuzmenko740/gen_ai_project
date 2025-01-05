import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.base import RunnableBinding
from langchain.chains.combine_documents import create_stuff_documents_chain

from pathlib import Path

FAISS_INDEX_PATH = str(Path().absolute()) + "rag_app/faiss_index"


class RagChain:
    def __init__(self, api_key:str):
        load_dotenv()
        self.OPENAI_API_KEY = api_key
        self.vector_db = self.load_vector_db()
        self.llm = self.initialize_llm_model()
        self.history_aware_retriever = self.initialize_history_aware_retriever()
        self.question_answer_chain = self.initialize_question_answer_chain()

    def load_vector_db(self) -> FAISS:
        embeddings = OpenAIEmbeddings(api_key=self.OPENAI_API_KEY)
        vector_db = FAISS.load_local(
            str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )
        return vector_db

    def initialize_llm_model(self) -> ChatOpenAI:
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0125")
        return llm

    def initialize_history_aware_retriever(self) -> RunnableBinding:
        retriever = self.vector_db.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user input "
            "which might reference context in the chat history, "
            "formulate a standalone input which can be understood "
            "without the chat history. Do NOT answer the user input, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )
        return history_aware_retriever

    def initialize_question_answer_chain(self) -> RunnableBinding:
        system_prompt = (
            "You are a helpful and creative recipe generator. "
            "User gives you their products and their preferences. You can add products to complete the recipe. "
            "If you don't know the recipe of the dish, don't suggest it. "
            "You can ONLY give cooking-related advice. If a user asks something that is not related to cooking: Sorry, I can't reply to that. "
            "Use the following pieces of retrieved context to answer "
            "the user prompt.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        return question_answer_chain

    def get_rag_chain(self):
        return create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)


