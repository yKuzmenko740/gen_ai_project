
# Recipe Generator Application

**Recipe Generator Application** is an AI-powered tool that uses Retrieval-Augmented Generation (RAG) to suggest recipes based on user-provided ingredients and preferences. It integrates a conversational chatbot with a vector database to deliver accurate and personalized recipe recommendations.

## Features

- **Conversational Interface**: A user-friendly chat interface built with Streamlit.
- **Ingredient-Based Recipe Suggestions**: Enter ingredients, and the model will suggest recipes.
- **Dietary and Cooking Preferences**: Tailored suggestions based on time, complexity, or dietary restrictions.
- **Retrieval-Augmented Generation**: Enhanced context-awareness using a vector database for accurate recipe suggestions.

---

## Files and Their Purpose

### 1. `app.py`
- Entry point of the application.
- Contains a Streamlit-based web interface for interacting with the chatbot.
- Manages user chat history and communication with the backend chatbot.

### 2. `chat_model.py`
- Defines the `ChatBot` class.
- Handles user session management and passes user queries to the RAG model.
- Integrates message history and handles responses from the conversational chain.

### 3. `rag_model.py`
- Implements the `RagChain` class.
- Loads a FAISS vector database stored in the `faiss_index` folder.
- Configures the RAG pipeline, including:
  - Retrieval from the vector database.
  - Reformulation of user queries to maintain context.
  - A question-answer chain powered by OpenAI GPT models.
- Includes a custom prompt for the model to stay focused on recipe generation.

### 4. `faiss_index/`
- Contains the vector database used for retrieving contextual information.
- **Note**: This folder is necessary for running the application but is not included here. Ensure it's present in the correct directory.

---

## Installation

### Prerequisites
- Python 3.8 or higher.
- An OpenAI API key.

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/recipe_generator.git
   cd recipe_generator/rag_app
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the `rag_app` directory.
   - Add the following:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. **Ensure Vector Database is Present**:
   - Place the `faiss_index` folder inside `rag_app/`.

---

## Running the Application

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and go to:
   ```
   http://localhost:8501
   ```

3. Interact with the chatbot by entering questions like:
   ```
   What can I cook with chicken, garlic, and tomatoes?
   ```

---

## How It Works

1. **User Interaction**: Input your recipe query through the web interface.
2. **Message Handling**: `chat_model.py` processes your query, maintaining the conversation context.
3. **Retrieval and Generation**: 
   - `rag_model.py` retrieves relevant data from the FAISS vector database.
   - The OpenAI GPT model generates a contextual, recipe-based response.
4. **Response Delivery**: The chatbot provides recipe suggestions and cooking instructions.

---

## Notes

- The application strictly focuses on recipe generation. For unrelated queries, it politely declines assistance.
- Ensure the `faiss_index` folder is correctly loaded; otherwise, retrieval functionalities won't work.
- For customization, modify prompts in `rag_model.py`.

---
