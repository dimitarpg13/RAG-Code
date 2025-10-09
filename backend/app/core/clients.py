import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

# The load_dotenv function is going to load the .env file and capture as environment variables the constants defined within it.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# - Make sure to install the python-dotenv package.
# - Instantiate the AsyncOpenAI module with your API key.
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# TODO: Wrap your current OpenAI client to add observability by using the 
# wrap_openai function.
async_openai_client_obs = None

# TODO: Instantiate the Pinecone module with your API key.
pinecone_client = None