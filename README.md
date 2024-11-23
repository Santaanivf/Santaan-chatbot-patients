# Santaan Chatbot

Welcome to the **Santaan Chatbot** repository. This project was developed for **Santaan Startup** and serves as an intelligent medical assistant, designed to provide precise, document-based responses to user queries. Leveraging advanced AI technologies, this chatbot ensures accurate and contextually relevant interactions.

---

## Features
- Provides contextual answers by extracting information from medical resources.
- Utilizes state-of-the-art language models for high-quality response generation.
- Employs search with embeddings and vector storage for efficient query handling.
- Includes a user-friendly interface powered by Streamlit.
- Designed for scalability and adaptability in medical assistant applications.

---

## Project Structure
- **`chunker.py`**: Responsible for loading PDF files and splitting them into smaller, manageable text chunks for processing.
- **`embedder.py`**: Initializes the Hugging Face embeddings model for vector representation of the text chunks.
- **`vdb.py`**: Handles Pinecone vector store operations, including indexing and performing similarity searches on embeddings.
- **`chatbot.py`**: Implements the chatbot’s Streamlit interface, integrating document search and response generation functionalities.
- **`requirements.txt`**: Lists all dependencies necessary to set up and run the project.

---

## How to Use This Repository
1. **Clone the Repository**  
   Clone the repository using the following command:  
   ```bash
   git clone https://github.com/your-repo/santaan-chatbot.git
   cd santaan-chatbot
   ```
2. **Install Dependencies**
   Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Environment Variables**
   Create a .env file in the root directory and include the following:
   ```bash
   PINECONE_API_KEY=your_pinecone_api_key
   GROQ_API_KEY=your_groq_api_key
   ```
4. **Run the Application**
   Start the Streamlit application using the command:
   ```bash
   streamlit run chatbot.py
   ```
5. **Interact with the Chatbot**
   Open the URL provided by Streamlit in your browser and start chatting! 💬

---

## 👩‍💻 Contributor
Developed with ❤️ by **Dharsini Sri Balasubramaniam**  
📧 [Email](mailto:purpleunalome@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/dharsinisrib) | 🔗 [GitHub](https://github.com/dhars1n1)



