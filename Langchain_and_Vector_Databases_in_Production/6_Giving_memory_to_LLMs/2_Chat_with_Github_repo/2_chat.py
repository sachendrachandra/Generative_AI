from dotenv import load_dotenv, dotenv_values
import os

# Load environment variables from .env file
config = dotenv_values("C:/Users/SACHENDRA/Documents/Activeloop/.env")
load_dotenv("C:/Users/SACHENDRA/Documents/Activeloop/.env")

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "sachendra"
my_activeloop_dataset_name = "langchain_course_chat_with_gh"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
# Create a retriever from the DeepLake instance
retriever = db.as_retriever()

# Set the search parameters for the retriever
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["k"] = 10

# Create a ChatOpenAI model instance
model = ChatOpenAI()

# Create a RetrievalQA instance from the model and retriever
qa = RetrievalQA.from_llm(model, retriever=retriever)

# Return the result of the query
qa.run("What is the repository's name?")


import streamlit as st
from streamlit_chat import message

# Set the title for the Streamlit app
st.title(f"Chat with GitHub Repository")

# Initialize the session state for placeholder messages.
if "generated" not in st.session_state:
	st.session_state["generated"] = ["i am ready to help you ser"]

if "past" not in st.session_state:
	st.session_state["past"] = ["hello"]

# A field input to receive user queries
# user_input = st.text_input("", key="input")

# A field input to receive user queries
user_input = st.text_input("Enter your query:", key="input")

# Search the databse and add the responses to state
if user_input:
	output = qa.run(user_input)
	st.session_state.past.append(user_input)
	st.session_state.generated.append(output)

# # Create the conversational UI using the previous states
# if st.session_state["generated"]:
# 	for i in range(len(st.session_state["generated"])):
# 		message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
# 		message(st.session_state["generated"][i], key=str(i))

# Create the conversational UI using the previous states
if "generated" in st.session_state:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
"""
Use Below command to run the chatbot:
streamlit run ./chat.py
"""