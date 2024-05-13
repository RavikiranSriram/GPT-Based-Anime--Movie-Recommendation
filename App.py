import pandas as pd
import streamlit as st
import tiktoken
import lancedb
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import LanceDB
from recommendation-systems import 

# Define custom prompt
template = """You are a movie recommender system that help users to find anime that match their preferences. 
Use the following pieces of context to answer the question at the end. 
For each question, suggest three anime, with a short description of the plot and the reason why the user migth like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Your response:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=docsearch.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)

# Streamlit app
st.title('Anime Recommendation System')

query = st.text_input('Enter your query:', '')

if query:
    with st.spinner('Finding recommendations...'):
        with get_openai_callback() as cb:
            result = qa_chain({"query": query})
        
        recommendations = result['result']
        st.subheader('Top 3 Recommendations:')
        for i, anime in enumerate(recommendations):
            st.write(f"**Recommendation {i+1}:**")

