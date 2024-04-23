import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dotenv import load_dotenv
import os

load_dotenv()

# API Keys - Loaded from the .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Prompt Templates

general_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

general_template_with_history = """Use the following pieces of information to answer the user's question. Use the history to answer the question if you can.
Chat History:
{chat_history}
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

common_template = """
You are a helpful AI assistant. Provide the answer for the following question:

Question: {question}
Answer:
"""


# Model Paths - all from huggingface
LLAMA_PATH = "meta-llama/Llama-2-7b-chat-hf"
FINETUNE_PATH = "0xhzx/nv-qa"
PDF_CHAT_PATH = "gpt-3.5-turbo-1106"

