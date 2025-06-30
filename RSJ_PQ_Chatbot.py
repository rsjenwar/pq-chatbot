# (C) Rajeshwar Singh Jenwar rsjenwar@gmail.com 1-DEC-23
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import asyncio
import time
import os

import dotenv
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.sitemap import SitemapLoader
import hashlib
import logging
from langchain.callbacks import get_openai_callback
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
from PIL import Image
from streamlit_feedback import streamlit_feedback
from langchain_openai import AzureChatOpenAI



# App page icon and title in the browser, MUST be first command of the app
img=Image.open("data/images/digital_india_logo_1.png")
st.set_page_config(page_title="PQChatbot", page_icon=img)

logging.basicConfig(filename='logs/app.log', format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', encoding='utf-8', level=logging.INFO)


LLM_MODELS = {
    "deepseek-r1",
    "llama3",
    "llama3.2"
}
LLM_MODEL = "deepseek-r1"
# os.environ["STREAMLIT_FORCE_SINGLE_THREAD"] = 'True' #working

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Parliament Questions AI Chatbot')
    st.markdown('''
    ## About
    This Generative AI App can generate answers for parliament questions based on answers to similar questions given in past by the Ministry of Electronics and IT (MEITY), Government of India. The AI Engine is trained on both Lok Sabha and Rajya Sabha question-answer pairs pertaining to MEITY, from year 2014 onwards till last session. 
    
    ''')
    add_vertical_space(1)
   #  st.markdown('''
   # ## Large Language Model (LLM)
   # ''')
# LLM_MODEL = st.sidebar.selectbox("Select your preferred LLM",
  #                                   list(LLM_MODELS))
   # if LLM_MODEL == "llama2":
    #    LLM_MODEL = "ParlGPTllama2"
    st.markdown('''
    ## Technologies
    This Generative AI App is built using following purely open, secure and free technologies:
    - [LLAMA4](https://www.llama.com/models/llama-4/) Large Language Model (LLM)
    - [LangChain](https://python.langchain.com/)
    ''')
    add_vertical_space(1)

    st.markdown('''
    ## Creator
    - Rajeshwar Singh Janwar (RSJ@MEITY.GOV.IN)
    ''')
    # st.write('Made with ❤️ by Rajeshwar Singh Janwar (RSJ@MEITY.GOV.IN)')

load_dotenv()

import re


## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe and formal. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """You are a helpful, respectful and honest assistant for the use of Government's departments. Always answer in safe, formal and similar style of language as used in the following context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""

from langchain.prompts import PromptTemplate
prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": llama_prompt}

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text
import re
def remove_prefix(string, prefix):
    return re.sub(f'^{prefix}', '', string)

def extract_pq_number(file_name):
    # Pattern: 'AU' or 'AS', then digits, then either '_' or '.pdf'
    match = re.search(r'(?:AU|AS)(\d+)(?:_|\.pdf)', file_name, re.IGNORECASE)
    if match:
        return match.group(1)  # The number as a string
    else:
        return None

def print_parl_question_references(llm_response):
    n = 0
    pages = None
    dated = None
    st.write("\n __:orange[Reference(s):]__ \n")
    for source in llm_response["source_documents"]:
        n += 1
        logging.debug("[RSJ | DEBUG] \n Source.page_content repr: %s", {repr(source.page_content)})
        file_name = str(source.metadata['source'])
        parl = "Lok Sabha" if (file_name.__contains__("_loksabhaquestions_")) else "Rajya Sabha"
        question_type = "Unstarred" if (file_name.__contains__("_AU")) else "Starred ⭐"
        question_no = extract_pq_number(file_name)
        question_no_star = "⭐" if(question_type.__contains__("Starred")) else ""
        date = re.search(r'TO BE ANSWERED ON:? \d{1,2}.\d{1,2}.\d{4}', source.page_content)
        logging.debug("[RSJ | DEBUG] \n date: %s", {date})
        if date:
            dated = remove_prefix(date.group(), 'TO BE ANSWERED ON:? ')
        else:
            # It is possible that we got a part vector store of pdf file, not having date info
            # then, load pdf file and extract date from 1st page
            pdf_loader = PyPDFLoader(file_name)
            pages = pdf_loader.load()
            logging.debug("[RSJ | DEBUG] \n pages[0].page_content repr: %s", {repr(pages[0].page_content)})
            date = re.search(r'TO BE ANSWERED ON:? \d{1,2}.\d{1,2}.\d{4}', pages[0].page_content)
            if date:
                dated = remove_prefix(date.group(), 'TO BE ANSWERED ON:? ')
        title = re.search(r'TO BE ANSWERED ON:? \d{1,2}.\d{1,2}.\d{4}.?.?\n.?\n?.?\n?(.*?)\n', source.page_content)
        logging.debug("[RSJ | DEBUG] \n title: %s", {title})
        question_title = None
        if title:
            question_title = remove_prefix(title.group(), 'TO BE ANSWERED ON:? \d{1,2}.\d{1,2}.\d{4}')
        else:
            if pages:
                title = re.search(r'TO BE ANSWERED ON:? \d{1,2}.\d{1,2}.\d{4}.?.?\n.?\n?.?\n?(.*?)\n', pages[0].page_content)
            else:
                pdf_loader = PyPDFLoader(file_name)
                pages = pdf_loader.load()
                title = re.search(r'TO BE ANSWERED ON:? \d{1,2}.\d{1,2}.\d{4}.?.?\n.?\n?.?\n?(.*?)\n', pages[0].page_content)
            if title:
                question_title = remove_prefix(title.group(), 'TO BE ANSWERED ON:? \d{1,2}.\d{1,2}.\d{4}')

        if question_title:
            question_title = question_title.strip()
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.write(f"\t [{n}]: Question Title: {question_title}, {parl}, {question_type}, Question No.: {question_no_star}{question_no}, Dated: {dated}")
        # st.write(f"\t[{n}] : {file_name}")
        # file_name = str(source.metadata['source'])
        with open(file_name, "rb") as f:
            with col2:
                st.download_button("⬇️ Download PDF",
                               data=f,
                               file_name=file_name,
                               mime="application/octet-stream",
                               key=n)
        # st.link_button(file_name, file_name)

def process_llm_response(llm_response):
    st.write("\n __:orange[Question:]__ \n")
    st.write(wrap_text_preserve_newlines(llm_response['query']))
    st.write("\n __:orange[Answer, generated by AI:]__ \n")
    st.write(wrap_text_preserve_newlines(llm_response['result']))
    # take user feedback
    feedback = streamlit_feedback(
        # feedback_type="faces",
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide your feedback and suggestions.",
        on_submit=submit_feedback,
        key=f"feedback_{st.session_state.feedback_key}",
        # align="flex-start"
    )
    # add_vertical_space(1)
    print_parl_question_references(llm_response)


def submit_feedback(feedback):
    # add_vertical_space(3)
    # st.write("\n __:orange[Feedback:]__\n")
    #face = feedback['score']
    #score = {"😀": 5, "🙂": 4, "😐": 3, "🙁": 2, "😞": 1}[face]
    thumb = feedback['score']
    score = { "👍": 1, "👎": 0} [thumb]
    comment = feedback['text'] or "none"
    logging.info("[RSJ | INFO] [User Feedback] Score: %s, Comment: %s, User Query: %s, AI Response: %s",
                  score, comment, st.session_state.response['query'], st.session_state.response['result'])
    st.success("Thank you for your feedback!🙏🏼")
    st.balloons()


async def create_persist_db(docs, embedding, dir):
    vector_dB = await Chroma.from_documents(documents=docs,
                                           embedding=embedding,
                                           persist_directory=dir)
    # time.sleep(15)
    vector_dB.persist()
    print(docs)
    return vector_dB


def RSJ_PQ_Chatbot():
    st.header("Parliament Questions Chatbot 💬:flag-in:")
    add_vertical_space(1)
    vector_db_initialized = 'false'
    pdf_folder_path = "data/PQ_MEITY"
    mnt_path = "/mnt/chroma"
    #pdf_folder_path="/Users/rsj/Documents/PQ_MEITY"

    if 'vector_db' not in st.session_state:
        MD5hash_pdf_persits_dir = os.path.join(mnt_path, hashlib.md5(pdf_folder_path.encode('utf-8')).hexdigest())
        #print("rsj MD5hash_pdf_persits_dir", MD5hash_pdf_persits_dir)
        #st.write("rsj MD5hash_pdf_persits_dir", MD5hash_pdf_persits_dir)
        if os.path.exists(MD5hash_pdf_persits_dir):
            vectordb = Chroma(persist_directory=MD5hash_pdf_persits_dir,
                                 embedding_function=HuggingFaceEmbeddings())
            logging.info("[RSJ | INFO] Loaded Vector Store from: %s  %s", MD5hash_pdf_persits_dir, pdf_folder_path)
        else:
            pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
            pdf_docs = pdf_loader.load()
            chunk_size = 700
            chunk_overlap = 50
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(pdf_docs)
            logging.debug("[RSJ | DEBUG] All documents splitted into %d chunks with chunk size %d and chunk_overlap %d", len(all_splits), chunk_size, chunk_overlap)
            #print("[RSJ | DEBUG] All documents splitted into %d chunks with chunk size %d and chunk_overlap %d", len(all_splits), chunk_size, chunk_overlap)
            #st.write("[RSJ | DEBUG] All documents splitted into %d chunks with chunk size %d and chunk_overlap %d", len(all_splits), chunk_size, chunk_overlap)
            vectordb = create_persist_db(all_splits,
                                         HuggingFaceEmbeddings(),
                                         MD5hash_pdf_persits_dir)
            vectordb = None
            # time.sleep(15)
            logging.debug("[RSJ | DEBUG] Created Document folder Vector Store %s %s",
                          MD5hash_pdf_persits_dir, pdf_folder_path)
            vectordb = Chroma(persist_directory=MD5hash_pdf_persits_dir,
                                embedding_function=HuggingFaceEmbeddings())
            logging.debug("[RSJ | DEBUG] Loaded Vector Store from persistent directory %s", MD5hash_pdf_persits_dir)
            #st.write("[RSJ | DEBUG] Loaded Vector Store from persistent directory %s", MD5hash_pdf_persits_dir)
        # assign in session state in first go itself
        st.session_state.vector_db = vectordb

    if st.session_state.vector_db:
        # Accept user questions/query
        # query = st.text_input(":orange[Ask questions about your data]")
        # st.write(query)
        query = st.text_area("\n __:orange[Please enter your question here:]__\n")
        if query:
            # retriever = vectordb.as_retriever()
            # docs = retriever.get_relevant_documents(query=query, search_kwargs={"k": 3})
            # docs = vectordb.similarity_search(query=query, k=3)
            if 'query' not in st.session_state:
                st.session_state.query = ""
                logging.debug("[RSJ | DEBUG] Initializing User Query into the  session: %s with LLM: %s", query, LLM_MODEL)
            if st.session_state.query != query:
                st.session_state.query = query
                # LLM

                #model = "deepseek-r1"
                #model ="llama3"
                #model ="llama2"
                #model = "ParlGPTllama2:latest"
                #model = "mistral"

                llm = AzureChatOpenAI(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    openai_api_type="azure",
                    azure_deployment= os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), #"gpt-4o-rsj-chatbot",  # Replace with your deployment name
                    api_version= os.getenv("AZURE_OPENAI_API_VERSION"), #"2025-01-01-preview",  # Use your API version
                    temperature=0,
                    #max_tokens=512,
                )
                #response = llm.invoke("What is weather in Delhi now?")
                #st.write(response)

                # llm = Ollama(model=LLM_MODEL,
                #              verbose=True,
                #              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

                # llm = Ollama(base_url="http://host.docker.internal:11434", model="llama2",
                #              verbose=True,
                #               callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

                ## logging.info("[RSJ | INFO] Loaded LLM model: %s for query: %s", llm.model, query)
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
                # chain = load_qa_chain(llm=llm, chain_type="stuff")
                # response = chain.run(input_documents=docs, question=query)
                qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                       chain_type="stuff",
                                                       retriever=retriever,
                                                       chain_type_kwargs=chain_type_kwargs,
                                                       return_source_documents=True)
                llm_response = qa_chain(query)
                logging.info("[RSJ | INFO] Got LLM response for query: %s", query)

                if 'response' not in st.session_state:
                    st.session_state.response = None
                if 'feedback_key' not in st.session_state:
                    st.session_state.feedback_key = 0
                    logging.debug("[RSJ | DEBUG] Initializing response into the  session: %s", query)
                if st.session_state.response != llm_response:
                    st.session_state.response = llm_response
                    st.session_state.feedback_key += 1
                # process LLM response for user question
                process_llm_response(llm_response)

                logging.info("[RSJ | INFO] LLM: Question: %s, Answer: %s", llm_response['query'], llm_response['result'])
            else:
                process_llm_response(st.session_state.response)


if __name__ == '__main__':
    RSJ_PQ_Chatbot()
