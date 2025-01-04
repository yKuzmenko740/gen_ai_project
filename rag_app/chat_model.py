from rag_model import RagChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class ChatBot:
    def __init__(self, api_key:str):
        self.store = {}
        self.api_key = api_key
        self.conversational_rag_chain = self.create_conversational_rag_chain()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def create_conversational_rag_chain(self) -> RunnableWithMessageHistory:
        rag_chain = RagChain(self.api_key).get_rag_chain()
        return RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_model_answer(self, prompt: str, session_id: str) -> str:
        return self.conversational_rag_chain.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        )["answer"]


if __name__ == "__main__":
    chatbot = ChatBot()
    session_id = "test_session"
    prompt = "What can I make with chicken and rice?I want to cook it under hour and it should be not spicy"
    answer = chatbot.get_model_answer(prompt, session_id)
    print("Answer:", answer)