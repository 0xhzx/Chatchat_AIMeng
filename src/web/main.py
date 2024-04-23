import os, sys

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


import re
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
import chainlit as cl
from langchain_community.embeddings import OpenAIEmbeddings
import yaml
import logging
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
import pinecone
from chainlit.types import AskFileResponse
from langchain.chat_models import ChatOpenAI
try:
    from modules.chat_agent import ChatAgent
    from modules.constants import *
    from modules.helpers import get_sources
    from modules.vector_db import VectorDB
    from modules.memory import memory
    from modules.helpers import clear_gpu_memory

except:
    from web.modules.chat_agent import ChatAgent
    from web.modules.constants import *
    from web.modules.helpers import get_sources
    from web.modules.vector_db import VectorDB
    from web.modules.memory import memory
    from web.modules.helpers import clear_gpu_memory





# Adding option to select the chat profile
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="LLama",
            markdown_description="Use the local LLM: **LLama-7b-chat-hf*.",
        ),
        cl.ChatProfile(
            name="Fintuned-flan-flan-t5-base",
            markdown_description="Use for Nvidia QA",
        ),
        cl.ChatProfile(
            name="PDF Chat",
            markdown_description="Use OpenAI API for **gpt-3.5-turbo-1106**.",
        ),
    ]



# chainlit code
@cl.on_chat_start
async def start():
    clear_gpu_memory()
    with open("web/config.yml", "r") as f:
        config = yaml.safe_load(f)
        print(config)
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile is not None:
        # process pdf
        if chat_profile.lower() == "pdf chat":
            config["llm_params"]["model"] = PDF_CHAT_PATH
            config["llm_params"]["model_type"] = "pdf_chat"
            await cl.Avatar(
                name="Chatbot",
                url='https://upload.wikimedia.org/wikipedia/commons/e/e6/Duke_University_logo.svg',
            ).send()
            files = None
            welcome_message = """Welcome to the PDF/TXT QA System! To get started:
            1. Upload a PDF or text file
            2. Ask a question about the file
            """
            vector_db = VectorDB(config)
            while files is None:
                files = await cl.AskFileMessage(
                    content=welcome_message,
                    accept=["text/plain", "application/pdf"],
                    max_size_mb=20,
                    timeout=180,
                ).send()
            file = files[0]
            print(file)
            msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
            await msg.send()
            # No async implementation in the Pinecone client, fallback to sync
            docsearch = await cl.make_async(vector_db.get_docsearch)(file)

            chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
                chain_type="stuff",
                retriever=docsearch.as_retriever(),
                memory=memory,
                return_source_documents=True,
            )
            # Let the user know that the system is ready
            msg.content = f"`{file.name}` processed. You can now ask questions!"
            await msg.update()

            cl.user_session.set("chain", chain)
            cl.user_session.set("pdf", "PDF")
        # normal chat
        else:
            if chat_profile.lower() == "llama":
                config["llm_params"]["model"] = LLAMA_PATH
                config["llm_params"]["model_type"] = "llama"
            elif chat_profile.lower() == "fintuned-flan-flan-t5-base":
                config["llm_params"]["llm_loader"] = "fintuned-flan-flan-t5-base"
                config["llm_params"]["model"] = FINETUNE_PATH
                config["llm_params"]["model_type"] = "flan-t5"
            else:
                pass

            chat_agent = ChatAgent(config)
            chain = chat_agent.qa_bot()
            model = config["llm_params"]["model"]
            msg = cl.Message(content=f"Starting the bot {model}...")
            await msg.send()
            msg.content = f"Hey, please feel free to ask me some questions!"
            await msg.update()

            cl.user_session.set("chain", chain)
            clear_gpu_memory()
    else:
        print("chat_profile is None")
        return None

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "Chatchat"}
    return rename_dict.get(orig_author, orig_author)


@cl.on_message
async def main(message):
    clear_gpu_memory()

    chain = cl.user_session.get("chain")
    is_pdf_task = cl.user_session.get("pdf")
    # res = await chain.acall(message.content)
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")
    try:
        answer = res["answer"]
    except:
        answer = res["result"]
    print(f"answer: {answer}")

    answer_with_sources, source_elements = get_sources(res, answer)
    if is_pdf_task != "PDF":
        match = re.search(r'Helpful answer:(.*)', answer_with_sources, re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
        else:
            final_answer = "No match found"
    else:
        final_answer = answer_with_sources
    
    print(final_answer)
    del chain
    clear_gpu_memory()

    await cl.Message(content=final_answer, elements=source_elements).send()
