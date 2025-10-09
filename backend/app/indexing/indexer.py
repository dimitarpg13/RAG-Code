import logging
import asyncio
import numpy as np
import uuid
from pathlib import Path
from collections import defaultdict
from pydantic import BaseModel, Field
from typing import Literal
from pinecone_text.sparse import SpladeEncoder, BM25Encoder, SparseVector
from pinecone import ServerlessSpec

from app.core.clients import async_openai_client, pinecone_client
from app.indexing.schemas import CodeElement

logger = logging.getLogger(__name__)

INDEX_NAME = "github-repo-index"

# TODO: implement your system prompt.
SYSTEM_PROMPT = None
FILTER_SYSTEM_PROMPT = None

class DocumentType(BaseModel):
    type: Literal['code', 'doc', 'both'] = Field(
        ..., description="Picker for Pinecone file-type filtering: 'code' → ['.py'], 'doc' → ['.md'], 'both' → ['.py', '.md']."
    )


class Indexer:

    # take the code element -> summarize
    # summary -> embed it
    # index the data with vector representation of the summary
    # ignore sparse encoding for now

    # search function
    # - filter search (ignore)
    # - rerank the results



    def __init__(self, owner, repo, ref, namespace) -> None:
        
        # TODO: Create a namespace for the repo. Namespaces can use alphanumeric and dash '-' characters.
        self.namespace = None
        if not pinecone_client.has_index(INDEX_NAME):
            # TODO: if the index does not exist, use the pinecone_client.create_index function to create it. 
            # - Even if we use hybrid search, we will need to define the vector_type as "dense"
            # - Pass the index name
            # - The default dimension of the vector coming from OpenAI's Embedding model is 1536. 
            # That will be the number for the dimension argument. 
            # - BM25 and SPLADE require a dot product similarity metric to retrieve the vectors. 
            # With dense vectors, it is usually better to use the cosine similarity metric. 
            # To recover the cosine similarity metric, we will need to normalize the vectors and use 
            # the "dotproduct" as the default similarity metric (in the metric argument).
            # - Pass spec=ServerlessSpec(cloud="aws", region="us-east-1").
            pass

        # TODO:  Instantiate the Pinecone index using the Index class.
        self.index = None

    async def summarize_element(self, code_element: CodeElement) -> str:
        """Generate a concise description for a code element using AI.
        
        Args:
            code_element: CodeElement object containing code/doc content to summarize
            
        Returns:
            String containing AI-generated summary text (or existing description if available)
            
        Note:
            Returns existing description if already set to avoid redundant API calls.
            Uses OpenAI's GPT model to create factual descriptions based on SYSTEM_PROMPT.
            Returns None on API errors (logs exception to stdout).
        """

        # TODO: Create the messages that will be passed to the input argument. 
        # The messages are a list of messages. A message is a dictionary with a role 
        # ("system", "assistant", "user", "tool") and a content (the text of the message). For example:
        #   {'role': 'system', 'content': SYSTEM_PROMPT}``
        # 
        # We need a message for the system prompt and a message to expose the CodeElement content. 
        # For the CodeElement message, you can use the role user. For the text of the message,
        #  you can dump the text of a Pydantic model by using:
        # `code_element.model_dump_json(indent=2, exclude_none=True)``
        messages = []

        try: 
            # TODO: In async_openai_client.responses.create, choose a model (e.g. model='gpt-4.1-nano'), 
            # and pass the messages to the input argument. I like 'gpt-4.1-nano' as it is cheap and fast. 
            # You can pass a temperature to the temperature argument. I like to choose temperature=0.1, 
            # to get close to deterministic results, but to allow the LLM not to be stuck in local minima.
            response = await async_openai_client.responses.create(...)
            return response.output_text
        except Exception as e:
            logger.error(f"Problem with summarizing: {str(e)}")

    async def summarize_batch(self, code_elements: list[CodeElement]) -> list[CodeElement]:
        """Generate AI summaries for a batch of code elements concurrently.
        
        Args:
            code_elements: List of CodeElement objects to summarize
            
        Returns:
            Same list of CodeElement objects with description field populated
            
        Note:
            Processes all elements in parallel using asyncio.gather().
            Handles exceptions gracefully - only updates description if AI call succeeds.
            Modifies input objects in-place by setting their description attribute.
        """

        # TODO: Create a list of tasks for all the code_elements using the create_task function. 
        # Await the response from all those tasks using the gather function.
        tasks = []
        descriptions = None
        # TODO: For all the retrieved descriptions, assign the text of the description 
        # to the description parameter in the related CodeElement instance.
        return code_elements
    
    async def summarize_all(self, code_elements: list[CodeElement], batch_size: int = 1000) -> list[CodeElement]:
        """Generate AI summaries for all code elements in batches.
        
        Args:
            code_elements: List of CodeElement objects to summarize
            batch_size: Number of elements to process per batch (default: 1000)
            
        Returns:
            Same list of CodeElement objects with description fields populated
            
        Note:
            Processes elements in batches to manage memory and API rate limits.
            Each batch is processed concurrently using summarize_batch().
            Modifies input objects in-place by setting their description attribute.
        """
        # TODO: Implement summarize_all. Iterate through batches of size
        # batch_size and call summarize_batch for each batch.
        return code_elements
    
    async def embed_batch(self,  batch: list[CodeElement]) -> list[list[float]]:
        """Generate embeddings for a batch of code elements using OpenAI.
        
        Args:
            batch: List of CodeElement objects with populated description fields
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
            
        Note:
            Uses OpenAI's text-embedding-3-small model to embed element descriptions.
            Assumes all elements have valid description attributes set.
        """
        # TODO: Implement the embed_batch function by using the 
        # embeddings.create function with the description of the CodeElement
        raise NotImplemented
    
    async def embed_all(self, code_elements: list[CodeElement], batch_size: int = 1000) -> list[list[float]]:
        """Generate embeddings for all code elements in batches sequentially.
        
        Args:
            code_elements: List of CodeElement objects with populated description fields
            batch_size: Number of elements to process per batch (default: 1000)
            
        Returns:
            List of embedding vectors for all input elements
            
        Note:
            Processes batches sequentially to avoid overwhelming the API.
            Relies on OpenAI client's internal concurrency for optimal performance.
        """
        embeddings = []
        # TODO: Implement the embed_all function. Iterate through all the code 
        # elements and send them in batches to the embed_batch function.
        return embeddings
    
    def bm25_encode(self, code_elements: list[CodeElement]) -> list[SparseVector]:
        """Generate BM25 sparse vectors for code elements using their text content.
        
        Args:
            code_elements: List of CodeElement objects to encode
            
        Returns:
            List of SparseVector objects containing BM25-encoded representations
            
        Note:
            Fits BM25 encoder on the corpus of all element texts, then encodes each document.
            Uses text field (not description) for encoding to preserve original content structure.
            Saves fitted BM25 parameters to backend/BM25_params/{namespace}.json for query encoding.
        """
        # TODO: Implement bm25_encode: 
        # - Instantiate the BM25Encoder from pinecone_text.
        # - Create a corpus of text documents using the original text value of the CodeElement. 
        # - Fit the corpus of text.
        # - Save the model parameters using the bm25.dump function. Use the namespace as a unique identifier for the encoder.
        # - Encode all documents and return the results
        raise NotImplemented
    
    def splade_encode(
        self,
        code_elements: list[CodeElement],
        max_characters: int = 1000,
        stride: int = 500,
        batch_size: int = 32 
    ) -> list[SparseVector]:
        """Generate SPLADE sparse vectors for code elements using efficient batched encoding.
        
        Args:
            code_elements: List of CodeElement objects to encode
            max_characters: Maximum characters per sliding window chunk (default: 1000)
            stride: Step size between chunk starts, creates overlap (default: 500)
            batch_size: Number of text chunks to encode per SPLADE batch (default: 32)
            
        Returns:
            List of SparseVector objects, one per input element with merged window vectors
            
        Note:
            Uses sliding windows to handle long texts, batched encoding for efficiency,
            and max-pooling to merge multiple windows per document into single vectors.
        """
        encoder = SpladeEncoder()
        
        def _create_windows(text: str) -> list[str]:
            """Split text into overlapping chunks using sliding window approach."""
            if not text: return []
            chunks = []
            for start in range(0, len(text), stride):
                text_chunk = text[start:start+max_characters].strip()
                if text_chunk: chunks.append(text_chunk)
            return chunks
            
        # Step 1: Create sliding windows for each document, tracking document IDs
        windows: list[tuple[int, str]] = []
        for doc_id, element in enumerate(code_elements):
            # Get windows or fallback to full text if no windows created
            element_windows = _create_windows(element.text) or ([element.text] if element.text.strip() else [])
            for window_text in element_windows:
                windows.append((doc_id, window_text))
                
        # Handle case where no valid windows exist
        if not windows:
            return [{"indices": [], "values": []} for _ in code_elements]

        # Step 2: Process windows in batches and merge vectors per document using max-pooling
        merged: list[dict[int, float]] = [defaultdict(float) for _ in code_elements]
        for i in range(0, len(windows), batch_size):
            print(i, len(windows))  # Progress tracking
            # Extract just the text from current batch of windows
            batch_texts = [window_text for _, window_text in windows[i:i+batch_size]]
            # Encode all texts in this batch at once
            vectors = encoder.encode_documents(batch_texts)
            
            # Merge each vector back to its corresponding document using max-pooling
            for (doc_id, _), vector in zip(windows[i:i+batch_size], vectors):
                for idx, val in zip(vector["indices"], vector["values"]):
                    # Max-pooling: take maximum value across windows for each index
                    merged[doc_id][idx] = max(val, merged[doc_id].get(idx, 0.0))

        # Step 3: Convert merged dictionaries to final sparse vector format
        output_vectors: list[SparseVector] = []
        for merged_dict in merged:
            if not merged_dict:
                # Empty document gets empty vector
                output_vectors.append({"indices": [], "values": []})
            else:
                # Sort indices and extract corresponding values
                indices, values = zip(*sorted(merged_dict.items()))
                output_vectors.append({"indices": list(indices), "values": list(values)})
        return output_vectors
    
    def encode_sparse_query(self, query: str, sparse_bm25: bool = True) -> SparseVector:
        """Encode a search query into a sparse vector for retrieval.
        
        Args:
            query: Search query text to encode
            sparse_bm25: If True, use BM25 encoder; if False, use SPLADE encoder (default: True)
            
        Returns:
            SparseVector object containing encoded query representation
            
        Note:
            For BM25: loads pre-fitted parameters from backend/BM25_params/{namespace}.json
            For SPLADE: uses default encoder without pre-fitting requirements
        """
        # TODO: Implement encode_sparse_query. The pinecone_text package provides a encode_queries 
        # function for both the SPLADE and BM25 encoders:
        # - If sparse_bm25 is true, we encode the query with BM25, and SPLADE otherwise
        # - For BM25, we need to load the parameters we learned during the encoding for the training corpus

        raise NotImplemented
    
    def _is_index_empty(self) -> bool:
        """Check if the index namespace is empty.
        
        Returns:
            True if namespace has no vectors, False otherwise
        """
        stats = self.index.describe_index_stats()
        count = stats.get("namespaces", {}).get(self.namespace, {}).get("vector_count", 0)
        return count == 0

    def _l2_normalize(self, vectors: list[list[float]], eps: float = 1e-12) -> list[list[float]]:
        """L2 normalize vectors to unit length.
        
        Args:
            vectors: List of embedding vectors to normalize
            eps: Small epsilon to avoid division by zero (default: 1e-12)
            
        Returns:
            List of L2-normalized vectors
            
        Note:
            Converts to numpy for efficient computation, then back to lists.
        """
        vectors = np.asarray(vectors, dtype=np.float32)        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) 
        vectors = vectors / np.maximum(norms, eps)         
        return vectors.tolist()
    
    async def index_data(
            self, 
            code_elements: list[CodeElement], 
            sparse_bm25: bool = True,
            batch_size: int = 100,
            alpha: float = 0.8
        ) -> None:
        """Index code elements into Pinecone with hybrid dense+sparse vectors.
        
        Args:
            code_elements: List of CodeElement objects to index
            sparse_bm25: If True, use BM25 for sparse vectors; otherwise use SPLADE (default: True)
            batch_size: Number of vectors to upsert per batch (default: 100)
            alpha: Weight for dense embeddings in hybrid search (default: 0.8)
            
        Note:
            Skips indexing if namespace already contains data.
            Filters out elements without text or descriptions.
            Uses hybrid weighting: dense * alpha + sparse * (1-alpha).
            Filters out oversized metadata (>35KB) to avoid Pinecone limits.
        """

        # TODO: Implement index_data
        # TODO: Avoid repopulating the index if the namespace is not empty. I am giving you _is_index_empty for simplicity.

        # TODO: Use the summarize_all function to summarize all the text. 
        # Make sure to filter elements for which the text element is empty.
        code_elements = None
        # TODO: Use the embed_all function to create a dense vector representation for every code element. 
        # Make sure to filter elements for which the description element is empty.
        dense_embeddings = None
        # TODO: Use the _l2_normalize function to normalize all the vectors.
        dense_embeddings = None

        # TODO: Use the bm25_encode or splade_encode to create the sparse embeddings.
        if sparse_bm25:
            sparse_embeddings = None
        else:
            sparse_embeddings = None

        for i in range(0, len(code_elements), batch_size):
            
            # TODO: Get the batch of data, dense embeddings, and sparse embeddings.
            batch = None
            batch_dense_embeddings = None
            batch_sparse_embeddings = None

            sparse_indices = None
            sparse_values = None
            # TODO: Compute the metadata
            metadata = None
            # TODO: Create unique IDs
            vector_ids = None

            # TODO: Weigh the dense vectors VS sparse vectors
            # batch_dense_embeddings[j] = np.array(batch_dense_embeddings[j]) * alpha).tolist()
            # sparse_values[j] = (np.array(sparse_values[j]) * (1 - alpha)).tolist()
            data = [{
                'id': vector_ids[j],
                'values': batch_dense_embeddings[j],
                'sparse_values': {'indices': sparse_indices[j], 'values': sparse_values[j]},
                'metadata': metadata[j]
            # Pinecone can only upsert metadata with max 40kB of data so I added this filter
            # to account for the weaknesses of our current parsing pipeline
            } for j in range(len(batch)) if len(str(metadata[j])) < 35000]

            try: 
                res = self.index.upsert(vectors=data, namespace=self.namespace)
            except Exception as e:
                logger.error(f"Problem with indexing: {str(e)}")

    async def get_search_filter(self, query: str) -> str:

        # TODO: Get the messages
        messages = []

        try: 
            # TODO: Pass the DocumentType to the text_format argument.
            response = await async_openai_client.responses.parse(...)
            return response.output_parsed.type
        except Exception as e:
            logger.error(e)

    async def search(
            self, 
            query: str, 
            max_results: int = 15, 
            with_filters: bool = True, 
            with_rerank: bool = True,
            sparse_bm25: bool = True,
        ) -> list[CodeElement]:
        """Search for relevant code elements using hybrid dense+sparse retrieval.
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return (default: 15)
            with_filters: If True, uses AI to filter by file type (.py/.md) (default: True)
            with_rerank: If True, uses Cohere rerank-3.5 for result reordering (default: True)
            sparse_bm25: If True, uses BM25 for sparse; if False, uses SPLADE (default: True)
            
        Returns:
            List of CodeElement objects ranked by relevance
            
        Note:
            Combines dense embeddings (OpenAI) with sparse vectors (BM25/SPLADE) for hybrid search.
            AI filter selects appropriate file types based on query intent.
            Reranking improves result quality using description field for semantic matching.
        """

        # TODO: In the search function we are going to implement filters if with_filters is true:
        # - Use the get_search_filter to determine what kind of filters we need. 
        # Based on the result of the function, we will have ".py", ".md", or both.
        # - In this specific case, the filters are on the "extension" key.
        # `filters = {"extension": {"$in": extensions}}``
        # - If with_filters is not true, the default value is an empty dictionary {}.
        filters = None

        # TODO: If with_rerank is true, populate the rerank dictionary, otherwise set it as None.
        rerank = None

        # TODO: Generate the dense and sparse embeddings from the text query and implement this query_dict dictionary. 
        query_dict = {}

        result = self.index.search(
            namespace=self.namespace, 
            query=query_dict,
            rerank=rerank,
        )

        docs = result["result"]['hits']
        data = [CodeElement.model_validate(doc['fields']) for doc in docs]
        return data
