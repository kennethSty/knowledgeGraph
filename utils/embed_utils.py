# Load model directly
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings

# Env Vars
load_dotenv("../graph_generation/keys.env") #in local ../graph_generation
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class GerMedBert:
    """
    This is a class which wraps the interaction with the GerMEDBert model from hf.
    Note: GerMedBERT/medbert-512 is a MaskedLM model which means it has an additional head ontop of the BERT
    architecture for masked language modelling. Loading this model with AutoModel means we load the weights of the underlying basic BERT.
    For an embedding I choose the CLS (first) embedding vector. To improve performance, one could finetune this model on the MS_Marco dataset.
    """
    def __init__(self, device):
        self.device = device
        self.model = AutoModel.from_pretrained("GerMedBERT/medbert-512",  add_pooling_layer=False, token = HF_TOKEN).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("GerMedBERT/medbert-512", token = HF_TOKEN)

    def embed(self, batch):
        input_token_batch = self.tokenizer(batch, padding = True, truncation = True, return_tensors="pt").to(self.device)
        input_batch_ids = input_token_batch["input_ids"].to(self.device)
        print(f"Inputs on device:{input_batch_ids.device}")

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**input_token_batch)
            last_hidden_state = outputs.last_hidden_state

        # Extract the embedding of the [CLS] token (first token)
        cls_embedding = last_hidden_state[:, 0, :]
        print("Shape of CLS embedding:", cls_embedding.shape)
        return cls_embedding.tolist()

class OpenAIEmbedd:
    def __init__(self, dimensions=None, model_name = "text-embedding-3-large"):
        if dimensions is not None and model_name != "text-embedding-3-large":
            print("Warning: embedding does not support different dimensions.")
            print(f"New Model: text-embedding-3-large with dimension {dimensions}")
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=dimensions,
                                                    openai_api_key=OPENAI_API_KEY)
        if dimensions is not None and model_name == "text-embedding-3-large":
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=dimensions,
                                                    openai_api_key=OPENAI_API_KEY)
            print(f"Embedding model: text-embedding-3-large, Dim: {dimensions}")
        else:
            self.embedding_model = OpenAIEmbeddings(model = model_name, openai_api_key=OPENAI_API_KEY)

    def embed(self, batch: list):
        embedding = self.embedding_model.embed_documents(batch)
        return embedding





