import os
from openai import AsyncOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from langsmith import wrappers

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
async_openai_client_obs = wrappers.wrap_openai(async_openai_client)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)