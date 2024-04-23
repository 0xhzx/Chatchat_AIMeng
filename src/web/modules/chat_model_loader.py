import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from langchain_community.chat_models import ChatOpenAI
from langchain.llms import CTransformers
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer
from langchain.llms import LlamaCpp
import torch
import transformers
import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl


class ChatModelLoader:
    def __init__(self, config):
        self.config = config
        self.hf_token = os.getenv("HF_TOKEN")
        self.prompt_template = """
        You are a helpful AI assistant. Provide the answer for the following question:
        Question: {question}
        Answer:
        """

    @cl.cache
    def load_chat_model(self):
        if self.config["llm_params"]["model"] != None:
            # load from hf
            model = self.config["llm_params"]["model"]
            tokenizer = AutoTokenizer.from_pretrained(model, token=self.hf_token, return_token_type_ids=False)
            streamer = TextStreamer(tokenizer, skip_prompt=True)
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                # low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                max_length=1000,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                token=self.hf_token,
            )
            # use_auth_token=True,

            llm = HuggingFacePipeline(
                pipeline=pipeline,
                model_kwargs={"temperature": 0},
            )
        else:
            raise ValueError("Invalid LLM Loader")
        return llm

        # if self.config["llm_params"]["llm_loader"] == "openai":
        #     llm = ChatOpenAI(
        #         model_name=self.config["llm_params"]["openai_params"]["model"]
        #     )
        # elif self.config["llm_params"]["llm_loader"] == "local_llm":
        #     n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        #     model_path = self.config["llm_params"]["local_llm_params"]["model"]
        #     llm = LlamaCpp(
        #         model_path=model_path,
        #         n_batch=n_batch,
        #         n_ctx=2048,
        #         f16_kv=True,
        #         verbose=True,
        #         n_threads=2,
        #         temperature=self.config["llm_params"]["local_llm_params"][
        #             "temperature"
        #         ],
        #     )
        # else:
        #     raise ValueError("Invalid LLM Loader")
        # return llm
        