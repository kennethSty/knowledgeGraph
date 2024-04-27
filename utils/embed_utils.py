# Load model directly
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import os

# Env Vars
load_dotenv("../graph_generation/keys.env")
HF_TOKEN = os.getenv("HF_TOKEN")


class GerMedBert:
    """
    This is a class which wraps the interaction with the GerMEDBert model from hf.
    Note: GerMedBERT/medbert-512 is a MaskedLM model which means it has an additional head ontop of the BERT
    architecture for masked language modelling. Loading this model with AutoModel means we load the weights of the underlying basic BERT.
    For an embedding I choose the CLS (first) embedding vector. To improve performance, one could finetune this model on the MS_Marco dataset.
    """
    def __init__(self, device):
        self.device = device
        self.model = AutoModel.from_pretrained("GerMedBERT/medbert-512",  add_pooling_layer=False, token = HF_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained("GerMedBERT/medbert-512", token = HF_TOKEN)

    def embed(self, batch):
        input_token_batch = self.tokenizer(batch, padding = True, truncation = True, return_tensors="pt")

        # Forward pass through the model
        outputs = self.model(**input_token_batch)
        last_hidden_state = outputs.last_hidden_state

        # Extract the embedding of the [CLS] token (first token)
        cls_embedding = last_hidden_state[:, 0, :]
        print("Shape of CLS embedding:", cls_embedding.shape)
        return cls_embedding


