  
# import streamlit as st
# from dotenv import load_dotenv
# import pickle

# from PyPDF2 import PdfReader
# from streamlit_extras.add_vertical_space import add_vertical_space
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
# from transformers import pipeline
# import os
 
# # Sidebar contents
# with st.sidebar:
#     st.title('🤗💬 LLM Chat App')

 
# load_dotenv()
 
# def main():
#     st.header("Chat with PDF 💬")

#        # upload a PDF file
#     pdf = st.file_uploader("Upload your PDF", type='pdf')
 
#     # st.write(pdf)
#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)
        
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
 
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#             )
#         chunks = text_splitter.split_text(text=text)
 
#         # # embeddings
#         store_name = pdf.name[:-4]
#         st.write(f'{store_name}')
#         # st.write(chunks)
 
#         if os.path.exists(f"{store_name}.pkl"):
#             with open(f"{store_name}.pkl", "rb") as f:
#                 VectorStore = pickle.load(f)
#             # st.write('Embeddings Loaded from the Disk')s
#         else:
#             embeddings = OpenAIEmbeddings()
#             VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#             with open(f"{store_name}.pkl", "wb") as f:
#                 pickle.dump(VectorStore, f)
 
#         # embeddings = OpenAIEmbeddings()
#         # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
#         # Accept user questions/query
#         query = st.text_input("Ask questions about your PDF file:")
#         # st.write(query)
 
#         if query:
#             docs = VectorStore.similarity_search(query=query, k=3)
 
#             llm = OpenAI(model_name='gpt-3.5-turbo')
#             chain = load_qa_chain(llm=llm, chain_type="stuff")
#             with get_openai_callback() as cb:
#                 response = chain.run(input_documents=docs, question=query)
#                 print(cb)
#             st.write(response)


 
# if __name__=='__main__':
#     main()
 
import streamlit as st
from dotenv import load_dotenv
import pickle

from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
# from transformers import pipeline
import os

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # Initialize conversation history using session_state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Store and load embeddings
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            # Add the user query to the conversation history
            st.session_state.conversation_history.append(f"User: {query}")

            # Perform the query and generate a response
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)

                # Add the bot's response to the conversation history
                st.session_state.conversation_history.append(f"Bot: {response}")
            
            # Display the conversation history in the app+
            conversation_text = "<br>".join(st.session_state.conversation_history)
            st.markdown(conversation_text, unsafe_allow_html=True)

if __name__=='__main__':
    main()


# import streamlit as st
# from dotenv import load_dotenv
# import pickle
# from PyPDF2 import PdfReader
# from streamlit_extras.add_vertical_space import add_vertical_space
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
# from transformers import pipeline
# import os

# # Sidebar contents
# with st.sidebar:
#     st.title('🤗💬 LLM Chat App')

# load_dotenv()

# def main():
#     st.header("Chat with PDF 💬")

#     # Initialize conversation history using session_state
#     if 'conversation_history' not in st.session_state:
#         st.session_state.conversation_history = []

#     # upload a PDF file
#     pdf = st.file_uploader("Upload your PDF", type='pdf')

#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)

#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text=text)

#         # Store and load embeddings
#         store_name = pdf.name[:-4]

#         if os.path.exists(f"{store_name}.pkl"):
#             with open(f"{store_name}.pkl", "rb") as f:
#                 VectorStore = pickle.load(f)
#         else:
#             embeddings = OpenAIEmbeddings()
#             VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#             with open(f"{store_name}.pkl", "wb") as f:
#                 pickle.dump(VectorStore, f)

#         # Accept user questions/query
#         query = st.text_input("Ask questions about your PDF file:")

#         if query:
#             # Add the user query to the conversation history
#             st.session_state.conversation_history.append(f"User: {query}")

#             # Perform the query and generate a response
#             docs = VectorStore.similarity_search(query=query, k=3)
#             llm = OpenAI(model_name='gpt-3.5-turbo')

#             # Generate a conversational response
#             bot_response = generate_conversational_response(llm, query)

#             # Add the bot's response to the conversation history
#             st.session_state.conversation_history.append(f"Bot: {bot_response}")

#             # Display the conversation history in the app
#             conversation_text = "<br>".join(st.session_state.conversation_history)
#             st.markdown(conversation_text, unsafe_allow_html=True)

# def generate_conversational_response(llm, query):
#     # Generate a conversational response using the GPT-3.5-turbo model
#     response = llm.generate([query], max_tokens=50, temperature=0.1, stop=None)
#     return response

# if __name__=='__main__':
#     main()
