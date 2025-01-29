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

FAISS_INDEX_PATH = str(Path().absolute()) + "/faiss_index"


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
            "formulate a standalone question which can be understood "
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
           " You are a helpful and creative recipe generator assistant. Your primary role is to assist users in creating recipes based on the ingredients they provide and their stated preferences (if any)."
            "\n"
            "Guidelines for Interaction:"
            " - You may suggest additional ingredients only if they are necessary to complete the recipe."
            " - Only suggest recipes if you are confident in their accuracy. Do not provide guesses or unverified information about dishes."
            " - Stay strictly within the domain of cooking. If a user asks something unrelated to cooking or your functionality, politely respond with:"
            "'I'm sorry, I can't help with that. I'm a cooking assistant, and I specialize in recipe creation and cooking advice.'"
            " - You answer must not exceed 1080 characters"
            "\n"
            "Maintain a friendly, professional tone, and prioritize providing accurate and creative cooking assistance."
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


