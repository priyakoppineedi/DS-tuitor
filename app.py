import streamlit as st
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Set up the page configuration for Streamlit app
st.set_page_config(page_title="Data Science AI Tutor", layout="wide")

# Custom CSS for styling the chat interface
st.markdown("""
    <style>
        .title-text { color: #4C82EF; font-size: 26px; font-weight: bold; text-align: center; }
        .user-message { background-color: #4C82EF; color: white; padding: 10px; border-radius: 10px; width: fit-content; margin-left: auto; }
        .ai-message { background-color: #E5E5E5; color: black; padding: 10px; border-radius: 10px; width: fit-content; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

# Sidebar controls for managing conversation
with st.sidebar:
    st.subheader("💬 Conversations")
    # Button to clear chat history
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []

# Display the title of the app
st.markdown('<div class="title-text">✨ Unlock the Power of Data Science with Your AI Tutor ✨</div>', unsafe_allow_html=True)
st.write("Welcome! Ready to dive into the world of Data Science? I'm here to help you learn and grow!")

# Load the API key for Google Generative AI
with open(r"D:/Docu/pyjunb/API_keys/API_key2_innomatics.txt") as f:
    key = f.read().strip()  # Replace this with a secure method to store the key
genai.configure(api_key=key)

# Define system-level instructions for the AI
system_prompt = """
You are an AI tutor specializing in Data Science. 
You only answer questions related to Data Science, including Machine Learning, Data Analysis, Python, SQL, and related topics. 
If asked about anything else, politely refuse and guide the user back to Data Science.

If the user greets you (e.g., 'hi', 'hello') at the **start** of the conversation, introduce yourself as an AI Data Science tutor. 
But do not keep repeating your introduction.

Provide **clear, informative, and concise** answers to all Data Science-related queries.

Example:
User: "What are some applications of data science?"
AI: "Data science is widely used in various industries. Some key applications include fraud detection in banking, recommendation systems in e-commerce, predictive analytics in healthcare, and customer sentiment analysis in marketing."

Stay engaging and helpful!
"""

# Set up the AI model and memory system
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=key)
memory = ConversationBufferMemory()

# Create the conversation prompt template
prompt_template = PromptTemplate(
    input_variables=["history", "input"], 
    template="{history}\nUser: {input}\nAI:"
)

# Set up the conversation chain using the LLM model and memory
conversation = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process user input
if user_input := st.chat_input("Type your question here..."):
    st.chat_message("user").markdown(user_input)  # Display user message in the chat
    st.session_state.messages.append({"role": "user", "content": user_input})  # Store user input in session state

    # Get the AI's response based on the conversation history
    response = conversation.run(history=memory.load_memory_variables({})["history"], input=user_input)

    with st.chat_message("assistant"):
        st.markdown(response)  # Display AI response in the chat

    # Store AI response in session state
    st.session_state.messages.append({"role": "assistant", "content": response})
