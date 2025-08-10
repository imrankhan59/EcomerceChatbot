import os
from dotenv import load_dotenv
import pandas as pd
from torch import embedding
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from src.data_converter import dataconverter

load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_ENDPOINT")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def ingest_data(mode):
    vstore = AstraDBVectorStore(
        embedding = embedding_model,
        collection_name = "ecommerce4",
        api_endpoint = ASTRA_DB_API_ENDPOINT,
        token = ASTRA_DB_TOKEN,
        namespace = ASTRA_DB_KEYSPACE
    )


    if mode == "create":
        docs = dataconverter()
        inserted_ids = vstore.add_documents(docs)
        return vstore, inserted_ids

    elif mode == "update":
        docs = dataconverter()
        inserted_ids = vstore.add_documents(docs)  # might upsert depending on settings
        return vstore, inserted_ids

    elif mode == "connect":
        return vstore

    else:
        raise ValueError("Invalid mode. Use 'connect', 'create', or 'update'.")
    

if __name__ == "__main__":
    pass
    #vstore, inserted_ids = ingest_data(None)
    #print(f"\n inserted {len(inserted_ids)} documents.")
    #result = vstore.similarity_search("can you tell me about budgest sound basshead")

    #for res in result:
        #print(f" {res.page_content} [{res.metadata}]")






