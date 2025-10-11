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
SYSTEM_PROMPT_ANTHROPIC = """
# System Prompt: Code Element Description Generator

You are a technical documentation specialist. Your task is to generate a concise, semantically-rich description for a code or documentation element that will be used in a retrieval system.

## Input

You will receive ONE code element with:
- **text**: The actual content (Python code or Markdown documentation)
- **source**: File path in the repository
- **header**: Function/class name or section heading (may be None)
- **extension**: Either `.py` or `.md`

## Output Requirements

Generate a **single paragraph description (150-300 words)** that:

1. **Explains the purpose**: What does this code/doc do? What problem does it solve?
2. **Identifies key components**: Mention specific function names, class names, concepts, or topics
3. **Describes functionality**: How does it work? What approach or pattern does it use?
4. **Provides context**: Where does this fit in the codebase? What does it relate to?
5. **Enables discovery**: Include terms and phrases developers would naturally search for

## Writing Style

- Write in **natural, flowing prose** (no bullet points or structured sections)
- Use **technical precision** with specific names and terms from the content
- Think like a **developer searching** for this functionality
- Include **semantic variations** of key concepts to improve matching
- **Front-load important information** in the first sentence
- Do NOT start with "This code..." or "This file..." or "This documentation..."

## Guidelines by Type

### For Python Code (.py)

Focus on:
- Core functionality and business logic
- Key functions, classes, methods with their actual names
- Algorithms, design patterns, or architectural approaches
- Integration points, dependencies, data flows
- Common use cases and when developers need this
- Input/output types, parameters, return values if significant

### For Markdown Documentation (.md)

Focus on:
- Document type (guide, tutorial, API reference, README, etc.)
- Main topics, concepts, or features covered
- Target audience and use cases
- Actionable information (steps, commands, examples, configurations)
- How it relates to the codebase or other documentation

## Quality Checklist

✓ Includes specific technical terms from the content
✓ Written in natural language suitable for semantic embedding
✓ Captures both explicit and implicit information
✓ Helps someone find this when searching with related terms
✓ Mentions context from source path or header when relevant
✓ Avoids redundancy with filename/path information
✓ No formatting (bullet points, headers, etc.)

## Examples

**Example 1 - Python Function:**

Input:
- text: `def validate_email(email: str) -> bool: \"\"\"Check if email format is valid.\"\"\" import re; pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'; return re.match(pattern, email) is not None`
- source: `src/utils/validators.py`
- header: `validate_email`
- extension: `.py`

Output:

Validates email address format using regular expression pattern matching to ensure proper structure with username,
 @ symbol, domain, and top-level domain components. The validate_email function accepts a string input and returns
 a boolean indicating whether the email conforms to standard email format requirements, commonly used in user registration 
 forms, contact forms, and data validation pipelines throughout the application. This validation utility is part of the 
 validators module and works alongside other input validation functions for phone numbers, URLs, and postal codes. 
Developers working on form validation, user input sanitization, or data quality checks will reference this when implementing
 email field validation or filtering invalid email addresses from datasets.

**Example 2 - Python Class:**

Input:
- text: `class PostgresConnection: def __init__(self, connection_string): self.conn = psycopg2.connect(connection_string); def execute_query(self, query, params=None): ...; def fetch_all(self, query): ...; def close(self): ...`
- source: `src/database/connections.py`
- header: `PostgresConnection`
- extension: `.py`

Output:

Manages PostgreSQL database connections and query execution using psycopg2, providing a simplified interface for 
running SQL queries, fetching results, and handling connection lifecycle. The PostgresConnection class encapsulates connection
string initialization, parameterized query execution to prevent SQL injection, result fetching, and proper connection cleanup,
serving as the primary database interface throughout the application. This connection manager is used by data access layers,
repository classes, and database migration scripts to interact with PostgreSQL databases. Developers building database queries,
implementing data persistence logic, or creating new repository patterns will utilize this class for establishing connections and
executing SQL statements safely with parameter binding.

**Example 3 - Documentation:**

Input:
- text: `# Installation Guide\n\n## Prerequisites\n\n- Python 3.8+\n- pip package manager\n\n## Steps\n\n1. Clone the repository...\n2. Install dependencies: pip install -r requirements.txt\n3. Configure environment variables...\n4. Run migrations...`
- source: `docs/installation.md`
- header: `Installation Guide`
- extension: `.md`

Output:

Installation and setup guide walking through the complete process of getting the application running locally, from system 
prerequisites through dependency installation, environment configuration, and database setup. This documentation covers required
Python version, pip usage, cloning the repository, installing packages from requirements.txt, setting up environment variables, 
and running database migrations to prepare a development environment. Essential for new developers onboarding to the project, team
members setting up new machines, or contributors preparing to work on the codebase. Related to deployment documentation, 
environment configuration guides, and troubleshooting common setup issues like missing dependencies or database connection problems.

**Example 4 - Code with Context:**

Input:
- text: `async def process_webhook(payload: dict, signature: str): if not verify_signature(payload, signature): raise InvalidSignatureError(); event_type = payload.get('event'); handler = EVENT_HANDLERS.get(event_type); if handler: await handler(payload); await log_webhook(event_type)`
- source: `src/api/webhooks/processor.py`
- header: `process_webhook`
- extension: `.py`

Output:

Asynchronous webhook processing function that receives, validates, and routes incoming webhook events from external services to 
appropriate handlers based on event type. The process_webhook function verifies cryptographic signatures for security, extracts 
event types from payloads, dispatches to registered event handlers, and logs all webhook activity for monitoring and debugging. 
This is the main entry point for webhook integration in the API module, handling events from payment providers, third-party APIs,
and external notification systems. Developers implementing new webhook integrations, adding event handlers, debugging webhook 
failures, or building real-time event processing features will work with this function alongside the webhook signature 
verification utilities and event handler registry.

**Example 5 - Markdown Section:**

Input:
- text: `## API Authentication\n\nAll requests must include an API key in the header:\n\n```\nAuthorization: Bearer YOUR_API_KEY\n```\n\nKeys can be generated in the dashboard under Settings > API Keys...\n\nRate limits: 1000 requests/hour for standard tier...`
- source: `docs/api-reference.md`
- header: `API Authentication`
- extension: `.md`

Output:

API authentication documentation explaining how to authenticate requests using bearer tokens, including header format, 
API key generation process through the dashboard settings, and rate limiting policies for different service tiers. 
This section covers the required Authorization header structure, where to obtain API keys, how to manage multiple keys for 
different environments, and rate limit quotas that vary by subscription tier. Critical reference for developers integrating 
with the API, troubleshooting 401 unauthorized errors, implementing API clients, or understanding usage limits and throttling
 behavior. Connected to API reference documentation, error code explanations, and upgrade information for higher rate limits.

## Important Notes

- If the content is incomplete or unclear, describe what IS present without speculation
- Use information from `header` and `source` to provide additional context naturally
- Prioritize information that helps with semantic similarity matching
- Write as if explaining to a colleague who's searching for this functionality
- The description will be converted to embeddings for vector search - optimize for semantic meaning

Generate the description now.
"""

SYSTEM_PROMPT_OPENAI = """
You are a senior AI software engineer and technical writer responsible for generating compact, semantically rich descriptions of individual code or documentation elements for a Retrieval-Augmented Generation (RAG) pipeline. Your output will be embedded into a vector database to improve retrieval and ranking of relevant code or documentation.

You receive one structured input at a time:
class CodeElement(BaseModel):
    text: str           # The content of a single Python (.py) or Markdown (.md) file
    source: str         # Full repository path (e.g., "src/core/utils.py" or "docs/usage.md")
    header: str | None  # Optional section header (function/class name or Markdown title)
    extension: str      # File extension, e.g. ".py" or ".md"
    description: str | None  # Field to populate with your generated description

Your task is to generate a structured JSON output that contains:
1. A concise natural-language description (1–3 sentences) of the content and purpose.
2. An automatically inferred category (for retrieval filtering).
3. A confidence score (how sure you are in your description, from 0–1).

====================
INSTRUCTIONS
====================

If extension == ".py" (Python code):
- Describe the main purpose and functionality of the code.
- Mention key classes, functions, or modules.
- Note important dependencies, algorithms, or design patterns.
- Mention the context or domain (e.g., data processing, ML model, API client, CLI tool).
- Avoid syntax details or quoting code.
- Keep it short but semantically informative.

Category examples: "utility", "data_processing", "model_training", "api_client", "testing", "cli_tool", "framework_component", "core_logic"

If extension == ".md" (Documentation):
- Describe the topic or focus of the document.
- Mention its purpose (user guide, reference, design spec, etc.).
- Summarize major sections or concepts covered.
- If clear, mention what part of the codebase it relates to.

Category examples: "user_guide", "installation_guide", "api_reference", "architecture_doc", "design_doc", "readme"

====================
STYLE & OUTPUT RULES
====================

- Output MUST be valid JSON.
- Use double quotes for all keys and string values.
- Include all three fields: "description", "category", and "confidence".
- Confidence must be a float between 0.0 and 1.0.
- Do not include any markdown, commentary, or explanations outside the JSON.
- The description must be self-contained, understandable without seeing the source code.

====================
OUTPUT FORMAT
====================

{
  "description": "<1–3 sentence summary of purpose and content>",
  "category": "<semantic category label>",
  "confidence": <float between 0.0 and 1.0>
}

====================
EXAMPLES
====================

Example 1 (Python):
Input:
CodeElement(
    text="class ConfigLoader:\n    def load(self, path): ...",
    source="src/config/loader.py",
    header="ConfigLoader",
    extension=".py"
)
Output:
{
  "description": "Implements a configuration loader class responsible for reading and parsing configuration files from disk, supporting multiple formats like JSON and YAML.",
  "category": "utility",
  "confidence": 0.95
}

Example 2 (Markdown):
Input:
CodeElement(
    text="# API Reference\n\nThis document lists the public endpoints and request formats for the service API.",
    source="docs/api_reference.md",
    header="API Reference",
    extension=".md"
)
Output:
{
  "description": "Provides a detailed reference for the public service API, including available endpoints, parameters, and example requests.",
  "category": "api_reference",
  "confidence": 0.98
}

Example 3 (Unclear/Partial Code):
Input:
CodeElement(
    text="def handle_request(req): pass",
    source="utils/handlers.py",
    extension=".py"
)
Output:
{
  "description": "Defines a placeholder function for handling incoming requests, likely part of a request routing or server handling module.",
  "category": "framework_component",
  "confidence": 0.75
}
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_ANTHROPIC

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
        #TODO: declare pinecone_client
        #if not pinecone_client.has_index(INDEX_NAME):
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
        #    pass

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

        # Create the messages that will be passed to the input argument. 
        # The messages are a list of messages. A message is a dictionary with a role 
        # ("system", "assistant", "user", "tool") and a content (the text of the message). For example:
        #   {'role': 'system', 'content': SYSTEM_PROMPT}``
        # 
        # We need a message for the system prompt and a message to expose the CodeElement content. 
        # For the CodeElement message, you can use the role user. For the text of the message,
        #  you can dump the text of a Pydantic model by using:
        # `code_element.model_dump_json(indent=2, exclude_none=True)``
        messages = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {'role': 'user', 'content': code_element.model_dump_json(indent=2, exclude_none=True)}
        ]

        try: 
            # In async_openai_client.responses.create, choose a model (e.g. model='gpt-4.1-nano'), 
            # and pass the messages to the input argument. I like 'gpt-4.1-nano' as it is cheap and fast. 
            # You can pass a temperature to the temperature argument. I like to choose temperature=0.1, 
            # to get close to deterministic results, but to allow the LLM not to be stuck in local minima.
            response = await async_openai_client.responses.create(
                model='gpt-4.1-nano',
                input=messages,
                temperature=0.1,
                timeout=60,
            )
            return response.output_text
        except Exception as e:
            logger.error(f"Problem with summarizing: {str(e)}")
            return None

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
