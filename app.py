import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

# Load environment variables
load_dotenv()

# Access your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the system and user templates
general_system_template =r""" 
You are a highly knowledgeable and experienced medical assistant AI. Your task is to assist users by providing medical advice based solely on the provided data from the 'Disease_symptom_and_patient_profile_dataset.csv'. 

- Ask relevant questions to better understand the patient's symptoms and condition.
- Provide possible diagnoses or recommendations based on the data available.
- If the data provided does not cover the user's query, suggest the user consult a healthcare professional.

Remember to always be cautious and prioritize user safety by encouraging consultation with a healthcare professional for any serious or uncertain situations.

----
{context}
----
"""
general_user_template = "Patient symptoms: ```{question}```"

messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# Function to read CSV data
def get_csv_text():
    # Load the CSV file
    csv_file = "Disease_symptom_and_patient_profile_dataset.csv"
    df = pd.read_csv(csv_file)
    # Convert the entire DataFrame to a single string
    text = df.to_string(index=False)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory, combine_docs_chain_kwargs={'prompt': qa_prompt})
    return conversation_chain

def handle_userinput(user_question, chat_container):
    if st.session_state.conversation is None:
        st.error(":red[Please process the documents first.]")
        return
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # Display user message
            chat_container.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # Display bot message
            chat_container.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with CSV Brain", page_icon="logo.png")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header(":blue[MediBot]")
    
    # Load and process the predefined CSV file at the start
    raw_text = get_csv_text()
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)
    
    # Container for chat history
    chat_container = st.container()  # This will precede the user input to ensure it's always at the bottom

    # Place the user input below the chat history
    user_question = st.text_input(":orange[Ask a question about your symptoms:]", key="user_input")
    
    # Update chat container with history upon receiving input
    if user_question:
        handle_userinput(user_question , chat_container)

if __name__ == '__main__':
    main()
