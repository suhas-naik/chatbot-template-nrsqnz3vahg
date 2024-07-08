{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2c69baca-8368-4411-92c9-e83e82c820fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Suvz/Library/Python/3.9/lib/python/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "from dotenv import load_dotenv\n",
    "from PyPDF2 import PdfReader\n",
    "from langchain.text_splitter import CharacterTextSplitter\n",
    "from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings\n",
    "from langchain.vectorstores import FAISS\n",
    "from langchain.chat_models import ChatOpenAI\n",
    "from langchain.memory import ConversationBufferMemory\n",
    "from langchain.chains import ConversationalRetrievalChain\n",
    "from htmlTemplates import css, bot_template, user_template\n",
    "from langchain.llms import HuggingFaceHub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "009e89e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_pdf_text(pdf_docs):\n",
    "    text = \"\"\n",
    "    for pdf in pdf_docs:\n",
    "        pdf_reader = PdfReader(pdf)\n",
    "        for page in pdf_reader.pages:\n",
    "            text += page.extract_text()\n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "33ba9a74",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_text_chunks(text):\n",
    "    text_splitter = CharacterTextSplitter(\n",
    "        separator=\"\\n\",\n",
    "        chunk_size=1000,\n",
    "        chunk_overlap=200,\n",
    "        length_function=len\n",
    "    )\n",
    "    chunks = text_splitter.split_text(text)\n",
    "    return chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "edd365b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_vectorstore(text_chunks):\n",
    "    #embeddings = OpenAIEmbeddings()\n",
    "    embeddings = HuggingFaceInstructEmbeddings(model_name=\"hkunlp/instructor-xl\")\n",
    "    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)\n",
    "    return vectorstore"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fb713158",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_conversation_chain(vectorstore):\n",
    "    llm = ChatOpenAI()\n",
    "    #llm = HuggingFaceHub(repo_id=\"google/flan-t5-xxl\", model_kwargs={\"temperature\":0.5, \"max_length\":512})\n",
    "\n",
    "    memory = ConversationBufferMemory(\n",
    "        memory_key='chat_history', return_messages=True)\n",
    "    conversation_chain = ConversationalRetrievalChain.from_llm(\n",
    "        llm=llm,\n",
    "        retriever=vectorstore.as_retriever(),\n",
    "        memory=memory\n",
    "    )\n",
    "    return conversation_chain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e1d23624",
   "metadata": {},
   "outputs": [],
   "source": [
    "def handle_userinput(user_question):\n",
    "    response = st.session_state.conversation({'question': user_question})\n",
    "    st.session_state.chat_history = response['chat_history']\n",
    "\n",
    "    for i, message in enumerate(st.session_state.chat_history):\n",
    "        if i % 2 == 0:\n",
    "            st.write(user_template.replace(\n",
    "                \"{{MSG}}\", message.content), unsafe_allow_html=True)\n",
    "        else:\n",
    "            st.write(bot_template.replace(\n",
    "                \"{{MSG}}\", message.content), unsafe_allow_html=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "08a3ca14",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-07-08 20:51:50.034 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Users/Suvz/Library/Python/3.9/lib/python/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-07-08 20:51:50.035 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    load_dotenv()\n",
    "    st.set_page_config(page_title=\"ChatKB\",\n",
    "                       page_icon=\":books:\")\n",
    "    st.write(css, unsafe_allow_html=True)\n",
    "\n",
    "    if \"conversation\" not in st.session_state:\n",
    "        st.session_state.conversation = None\n",
    "    if \"chat_history\" not in st.session_state:\n",
    "        st.session_state.chat_history = None\n",
    "\n",
    "    st.header(\"Chat with multiple KBs :books:\")\n",
    "    user_question = st.text_input(\"Ask a question about product:\")\n",
    "    if user_question:\n",
    "        handle_userinput(user_question)\n",
    "\n",
    "    with st.sidebar:\n",
    "        st.subheader(\"Your documents\")\n",
    "        pdf_docs = st.file_uploader(\n",
    "            \"Upload your PDFs here and click on 'Process'\", accept_multiple_files=True)\n",
    "        if st.button(\"Process\"):\n",
    "            with st.spinner(\"Processing\"):\n",
    "                # get pdf text\n",
    "                raw_text = get_pdf_text(pdf_docs)\n",
    "\n",
    "                # get the text chunks\n",
    "                text_chunks = get_text_chunks(raw_text)\n",
    "\n",
    "                # create vector store\n",
    "                vectorstore = get_vectorstore(text_chunks)\n",
    "\n",
    "                # create conversation chain\n",
    "                st.session_state.conversation = get_conversation_chain(\n",
    "                    vectorstore)\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b42bbbac",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
