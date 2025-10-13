import uuid
import logging
import numpy as np
from pathlib import Path
from typing import Literal
from app.core.clients import async_openai_client, pinecone_client
from app.indexing.schemas import CodeElement
from pinecone_text.sparse import SpladeEncoder, BM25Encoder, SparseVector
from pydantic import BaseModel, Field
from pinecone import ServerlessSpec
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


INDEX_NAME = "github-repo-index"


SYSTEM_PROMPT = """
**Role & goal**  
You are **ChunkDescriber**: a precise, non-speculative summarizer for RAG indexing. You receive a chunk of content that is either **code** or **documentation** and must return a concise, factual description. The summary will be embedded, so keep it dense, informative, and free of fluff.

## Inputs (variables)
- **KIND** — `".py"` or `".md"`.
- **TEXT** — exact chunk text.
- **PATH** — repo-relative file path.
- **HEADER** — optional context (imports/front-matter/breadcrumbs); may be empty.

## Primary task
- If **KIND = ".py"**: describe what the code does **at runtime**.
  Focus on: purpose; inputs/outputs; side effects (filesystem/network/db/stdout/logging/random/time/env/global state/concurrency); external APIs/libraries; notable control flow (retry/caching/memoization/error handling); invariants/constraints.  
  If tests, summarize the behavior/spec asserted. If partial, say so and describe only what's visible.

- If **KIND = ".md"**: describe what the text explains or specifies.
  Focus on: purpose & audience; key topics/sections; procedures/steps or workflows; commands/API endpoints/config flags/parameters; expected outcomes; prerequisites/assumptions; notable links/anchors; important tables or code fences (languages).  
  Describe only what's visible.

## Strict rules
- **No speculation.** Only claim facts visible in TEXT/HEADER/TITLE. If unknown, say “unknown” or omit.
- **No line-by-line narration** or pseudocode. Prefer compact, declarative summaries.
- **Do not invent** types, effects, or claims not evidenced in the chunk.
- **No chain-of-thought.** Provide conclusions only.
- Keep the **summary ≤ 200 words** (up to 250 if unusually complex).
- Use domain terms as written (e.g., *S3*, *SQLAlchemy*, *requests*, *NumPy*, *kubectl*).

## Tone and style Neutral, technical, terse. No marketing language. No hypotheticals beyond what the code shows.
"""

FILTER_SYSTEM_PROMPT = """
You are **FilterSelector**, a strict router that sets `DocumentType.type` to choose the best file-type filter for retrieval.

## Goal
Given a user query, return exactly one of:
- **'code'** → prioritize Python source files (maps to ['.py'])
- **'doc'**  → prioritize Markdown docs (maps to ['.md'])
- **'both'** → include both when the query likely needs code and docs, or is ambiguous (maps to ['.py', '.md'])

The runtime will map your choice to the Pinecone filter; you **must only** choose the label.

## Inputs
- `query`: the raw user query string (may include typos).

## Decision rules
1) Choose **'code'** when the query strongly targets implementation details or APIs:
   - Mentions functions/methods/classes or code tokens: “forward”, “__init__”, “def”, “class”, “args”, “return”.
   - Debugging/behavior/stack traces: “TypeError”, “AttributeError”, “why does this crash”.
   - Code usage inside scripts: “how to call”, “example code”, “unit test”, “refactor”.

2) Choose **'doc'** for conceptual/usage/overview/installation material:
   - “explain”, “overview”, “installation”, “configuration”, “prerequisites”, “limitations”, “tutorial”, “README”, “guide”, “FAQ”, “changelog”.

3) Choose **'both'** when:
   - The query mixes concept + implementation (e.g., “how attention masking works and how to call it”).
   - The query is short or ambiguous (“Llama tokenizer”, “adapter config”).
   - You can't confidently decide after applying 1-2.

### Tie-breakers
- Names a **specific function/method/class** → **'code'**.
- Names a **README / doc section** → **'doc'**.
- If still uncertain → **'both'**.

## Output format (STRICT)
Return **only** a JSON object valid for the Pydantic model:
{"type": "code" | "doc" | "both"}

No extra fields, no prose, no code fences.

## Examples
- Query: "Llama forward function"
  → {"type":"code"}
  Reason: Mentions a function; implementation likely in Python code.

- Query: "Install and configure the Llama repo"
  → {"type":"doc"}
  Reason: Installation/config are in docs/README.

- Query: "How does attention masking work and how to call it from my script?"
  → {"type":"both"}
  Reason: Needs concept (docs) and call pattern (code).

- Query: "README section on training settings"
  → {"type":"doc"}
  Reason: Explicitly asks for README section.

- Query: "TypeError in LlamaTokenizer.from_pretrained"
  → {"type":"code"}
  Reason: Stacktrace/debugging a Python API call.
"""

class DocumentType(BaseModel):
    type: Literal['code', 'doc', 'both'] = Field(
        ..., description="Picker for Pinecone file-type filtering: 'code' → ['.py'], 'doc' → ['.md'], 'both' → ['.py', '.md']."
    )


class Indexer:

    def __init__(self, owner=None, repo=None, ref=None, namespace=None) -> None:
        
        self.namespace = namespace or (f"{owner}-{repo}-{ref}" if ref else f"{owner}-{repo}")
        if not pinecone_client.has_index(INDEX_NAME):
            pinecone_client.create_index(
                name=INDEX_NAME,
                vector_type="dense",
                dimension=1536,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = pinecone_client.Index(INDEX_NAME)
        
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

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Code chunk:\n\n{code_element.model_dump_json(indent=2, exclude_none=True)}"}
        ]

        try: 
            response = await async_openai_client.responses.create(
                model='gpt-4.1-nano',
                input=messages,
                temperature=0.1,
                timeout=30.0
            )
            return response.output_text
        except Exception as e:
            print(e)

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

        tasks = [
            asyncio.create_task(self.summarize_element(element)) 
            for element in code_elements
        ]
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)
        for element, description in zip(code_elements, descriptions):
            if isinstance(description, str):
                element.description = description.strip()

        return code_elements
    
    async def summarize_all(self, code_elements: list[CodeElement], batch_size: int = 500) -> list[CodeElement]:
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
        for i in range(0, len(code_elements), batch_size):
            batch = code_elements[i:i+batch_size]
            await self.summarize_batch(batch)
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
  
        response = await async_openai_client.embeddings.create(
            input=[el.description for el in batch],
            model="text-embedding-3-small"
        )
        embeddings = [res.embedding for res in response.data]
        return embeddings
    
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
        for i in range(0, len(code_elements), batch_size):
            batch = code_elements[i:i+batch_size]
            batch_embeddings = await self.embed_batch(batch)
            embeddings.extend(batch_embeddings)
        return embeddings
    
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
        bm25 = BM25Encoder()
        corpus = [el.text for el in code_elements]
        bm25.fit(corpus)
        
        # Create BM25_params directory at the same level as app folder
        params_dir = Path(__file__).parent.parent.parent / "BM25_params"
        params_dir.mkdir(exist_ok=True)
        
        # Save with namespace name
        params_file = params_dir / f"{self.namespace}.json"
        bm25.dump(params_file)
        
        document_vectors = bm25.encode_documents(corpus)
        return document_vectors
    
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
        if sparse_bm25:
            encoder = BM25Encoder()
            params_file = Path(__file__).parent.parent.parent / "BM25_params" / f"{self.namespace}.json"
            encoder.load(params_file)
        else:
            encoder = SpladeEncoder()

        return encoder.encode_queries(query)

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

        if not self._is_index_empty():
            return
        
        code_elements = [el for el in code_elements if el.text]
        code_elements = await self.summarize_all(code_elements)
        code_elements = [el for el in code_elements if el.description]
        dense_embeddings = await self.embed_all(code_elements)
        dense_embeddings = self._l2_normalize(dense_embeddings)

        if sparse_bm25:
            sparse_embeddings = self.bm25_encode(code_elements)
        else:
            sparse_embeddings = self.splade_encode(code_elements)

        for i in range(0, len(code_elements), batch_size):
 
            batch = code_elements[i:i + batch_size]
            batch_dense_embeddings = dense_embeddings[i:i + batch_size]
            batch_sparse_embeddings = sparse_embeddings[i:i + batch_size]

            sparse_indices = [emb['indices'] for emb in batch_sparse_embeddings]
            sparse_values = [emb['values'] for emb in batch_sparse_embeddings]
            metadata = [el.model_dump(exclude_none=True) for el in batch]
            vector_ids = [str(uuid.uuid4()) for _ in batch]

            data = [{
                'id': vector_ids[j],
                'values': (np.array(batch_dense_embeddings[j]) * alpha).tolist(),
                'sparse_values': {
                    'indices': sparse_indices[j], 
                    'values': (np.array(sparse_values[j]) * (1 - alpha)).tolist()
                },
                'metadata': metadata[j]
            } for j in range(len(batch)) if len(str(metadata[j])) < 35000]

            try: 
                res = self.index.upsert(vectors=data, namespace=self.namespace)
            except Exception as e:
                logger.error(f"Problem with indexing: {str(e)}")

    async def get_search_filter(self, query: str) -> str:
        messages = [
            {'role': 'system', 'content': FILTER_SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Query: {query}"}
        ]

        try: 
            response = await async_openai_client.responses.parse(
                model='gpt-4.1-nano',
                input=messages,
                temperature=0.1,
                timeout=30.0,
                text_format=DocumentType
            )
            return response.output_parsed.type
        except Exception as e:
            print(e)
            
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

        if with_filters:
            document_type = await self.get_search_filter(query)
            extensions = ['.py', '.md']
            if document_type == 'code':
                extensions = ['.py']
            elif document_type == 'doc':
                extensions = ['.md']
            filters = {"extension": {"$in": extensions}}
        else:
            filters = {}

        if with_rerank:
            rerank = {
                "model": "cohere-rerank-3.5",
                "query": query,
                "top_n": max_results,
                "rank_fields": ["description"]
            }
        else:
            rerank = None

        response = await async_openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )

        dense_embedding = response.data[0].embedding
        sparse_embedding = self.encode_sparse_query(query, sparse_bm25)

        sparse_indices = []
        sparse_values = []
        if sparse_embedding:
            if isinstance(sparse_embedding, dict):
                sparse_indices = sparse_embedding.get('indices', [])
                sparse_values = sparse_embedding.get('values', [])
            else:
                sparse_indices = getattr(sparse_embedding, 'indices', [])
                sparse_values = getattr(sparse_embedding, 'values', [])

        use_sparse = bool(sparse_indices)

        vector_payload = {"values": dense_embedding}
        if use_sparse:
            vector_payload.update({
                "sparse_values": sparse_values,
                "sparse_indices": sparse_indices,
            })

        query_dict = {
            "vector": vector_payload,
            "top_k": max_results * 3 if with_rerank else max_results,
            "filter": filters,
        }

        result = self.index.search(
            namespace=self.namespace, 
            query=query_dict,
            rerank=rerank,
        )

        docs = result["result"]['hits']
        data = [CodeElement.model_validate(doc['fields']) for doc in docs]
        return data
    

        
    
