import streamlit as st
from vdb import docsearch
from groq import Groq
from dotenv import load_dotenv
import os

#Load environment variables from .env file
load_dotenv()


client = Groq(
    api_key="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"
)

# Streamlit app settings
st.set_page_config(
    page_title="Chatbot Interface",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Main app layout
st.markdown(
    """
    <style>
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
            margin-bottom: 15px;
        }
        .response-box {
            background-color: #F0F8FF;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            font-size: 1.1rem;
            color: #333;
        }
        .error {
            color: #FF4C4C;
            font-weight: bold;
            text-align: center;
        }
        .input-box {
            font-size: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Santaan Chatbot</div>', unsafe_allow_html=True)

# User input
user_question = st.text_input(
    "Type your question below:", 
    placeholder="Insert your question here", 
    key="input_box",
)

# Chatbot response
if user_question:
    try:
        # Perform document search
        r_docs = docsearch.similarity_search(query=user_question, k=2)
        context = "\n\n".join(doc.page_content for doc in r_docs)

        # System prompt for the chatbot
        system_prompt = "You are a medical assistant who answers patient questions using clear, simple, and detailed explanations based on trusted medical textbooks. Your responses should always be easy to understand, and in detail. Never give one line answer.  You don't answer anything outside textbook. You can identify questions even with spelling mistakes."

        # Get chatbot response
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""Answer the following question based on the context provided. 
                    If you do not know the answer, say "I do not know. DO NOT SAY based on the content when starting the answer." 
                    
                    ## Question:
                    {user_question}

                    ## Context:
                    {context}
                    """,
                },
            ],
            model="llama3-8b-8192",
        )

        response = chat_completion.choices[0].message.content

        # Display response
        st.markdown(
            f'<div class="response-box"><strong>Response:</strong><br>{response}</div>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.markdown(
            f'<div class="error">An error occurred: {str(e)}</div>',
            unsafe_allow_html=True,
        )
