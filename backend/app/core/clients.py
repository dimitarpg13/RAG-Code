import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

# The load_dotenv function is going to load the .env file and capture as environment variables the constants defined within it.
load_dotenv()

# TODO: 
# - Make sure to install the python-dotenv package.
# - Instantiate the AsyncOpenAI module with your API key.
async_openai_client = None

# TODO: Wrap your current OpenAI client to add observability by using the 
# wrap_openai function.
async_openai_client_obs = None

# TODO: Instantiate the Pinecone module with your API key.
pinecone_client = None