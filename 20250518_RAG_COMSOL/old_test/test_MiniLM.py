# test model


#%%

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [4, 4]
# %%
from huggingface_hub import snapshot_download

model_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_dir="./local_models/all-MiniLM-L6-v2", local_dir_use_symlinks=False)

# %%


#%%
from huggingface_hub import HfApi

from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

api = HfApi()
try:
    user_info = api.whoami(token=token)
    print("Token is valid! User info:", user_info)
except Exception as e:
    print("Token is invalid or expired:", e)

    
# %%
