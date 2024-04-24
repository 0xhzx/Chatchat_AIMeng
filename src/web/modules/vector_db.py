import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import chainlit as cl
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from modules.constants import *
import pinecone
from chainlit.types import AskFileResponse
from langchain.chat_models import ChatOpenAI

try:
    from modules.constants import *
    from modules.helpers import *
except:
    from constants import *
    from helpers import *


class VectorDB:
    '''
    currently only support Pinecone
    '''
    def __init__(self, config, logger=None):
        self.config = config
        self.index_name = config["embedding_options"]["index_name"]
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.namespaces = set()

    
    def process_file(self, file: AskFileResponse):
        if file.type == "text/plain":
            Loader = TextLoader
        elif file.type == "application/pdf":
            Loader = PyPDFLoader

        loader = Loader(file.path)
        documents = loader.load()
        docs = self.text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


    def get_docsearch(self, file: AskFileResponse):
        if file != None:
            docs = self.process_file(file)
            print(docs)
            # Save data in the user session
            cl.user_session.set("docs", docs)
            # Create a unique namespace for the file
            namespace = file.id
            if namespace in self.namespaces:
                docsearch = Pinecone.from_existing_index(
                    index_name=self.index_name, embedding=self.embeddings, namespace=namespace
                )
            else:
                docsearch = Pinecone.from_documents(
                    docs, self.embeddings, index_name=self.index_name, namespace=namespace
                )
                self.namespaces.add(namespace)
        else:
            docsearch = Pinecone.from_existing_index(
                    index_name=self.index_name, embedding=self.embeddings
            )
        return docsearch
    


    


if __name__ == "__main__":
    pass
