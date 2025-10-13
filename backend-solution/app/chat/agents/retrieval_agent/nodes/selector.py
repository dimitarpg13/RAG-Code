from pydantic import BaseModel, Field
from app.core.clients import async_openai_client
from app.chat.agents.retrieval_agent.state import RetrieverAgentState
from app.indexing.github_parsing import CodeElement
import logging
import asyncio

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a second-stage retrieval assistant.

Inputs you will receive for **each** call  
• **chat_history** - an ordered list of all messages exchanged so far,  
  each object having `role` (“user” | “assistant” | “system”) and `content`.  
  The last one or more entries with `role == "user"` constitute the
  current **user_query** you must focus on.  
• **doc_content** - the full text of ONE document returned by the first-stage
  embedding search (it may contain noise or tangential sections).

Your tasks  
1. Decide whether *doc_content* contains information that materially helps
   answer *user_question*.  
2. If it **does** help, extract only the parts that are most relevant
   (verbatim or lightly paraphrased) and pack them into the `extracted`
   field.  
3. If it **does not** help, leave `extracted` null.  
4. Provide a one-sentence rationale and your confidence.

Guidelines  
• Favour *precision*: mark `is_relevant = true` only when the document
  supplies concrete facts, definitions, procedures, or examples that the
  question requires.  
• Keep `extracted` short — ≤ 800 characters; include just the sentences,
  bullet points, or short code blocks that the downstream RAG step should
  quote or ground itself on.  
• If unsure, set `is_relevant = false` and confidence = "low".  
• Respond with **JSON that is valid for the DocFilterResult schema**; no
  markdown, no additional keys.
"""


class DocFilterResult(BaseModel):
    """
    Result of the second-stage filtering step for a single document.
    """
    is_relevant: bool = Field(
        ...,
        description="True if the document helps answer the user query, else False."
    )
    extracted: str | None = Field(
        None,
        description="Key snippets taken from the document "
                    "that directly address the question; null when is_relevant is False."
    )


class Selector:
    """Second-stage document selector that filters retrieved documents for relevance."""

    async def filter_doc(self, state: RetrieverAgentState, element: CodeElement) -> DocFilterResult:
        """Filter a single document for relevance to the chat history.
        
        Args:
            state: RetrieverAgentState containing chat_messages
            element: CodeElement to evaluate for relevance
            
        Returns:
            DocFilterResult with relevance decision and extracted content
            
        Raises:
            ConnectionError: If OpenAI API call fails
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"###  Chat_history  ###"},
        ]

        messages.extend(state.chat_messages[-10:])
        messages.append({"role": "user", "content": f"###  Document  ###\n\n{element.text}"})

        try:
            response = await async_openai_client.responses.parse(
                input=messages,
                # model='gpt-5-nano',
                # text={"verbosity": 'low'},
                # reasoning = {"effort": "minimal"},
                model='gpt-4.1-nano',
                temperature=0.1,
                text_format=DocFilterResult,
            )
            return response.output_parsed
        except Exception as e:
            logger.error(str(e))
            raise ConnectionError(f"Something wrong with Openai: {str(e)}")

    
    async def filter_documents(
            self, 
            state: RetrieverAgentState) -> list[DocFilterResult]:
        """Filter all new documents concurrently for relevance.
        
        Args:
            state: RetrieverAgentState containing new_documents to filter
            
        Returns:
            List of DocFilterResult objects for each document
        """
        
        tasks = [
            asyncio.create_task(self.filter_doc(state, doc)) 
            for doc in state.new_documents
        ]
        filters = await asyncio.gather(*tasks, return_exceptions=True)
        return filters
    
    async def __call__(self, state: RetrieverAgentState) -> RetrieverAgentState:
        """Main selector node - filters documents and updates retrieved_documents.
        
        Args:
            state: Current RetrieverAgentState
            
        Returns:
            Updated state with relevant documents added to retrieved_documents
        """

        filters = await self.filter_documents(state)
        # Extract only relevant document content
        for doc, filter in zip(state.new_documents, filters):
            if filter.is_relevant and filter.extracted:
                doc.text = filter.extracted
                state.retrieved_documents.append(doc)
        return state
    

selector = Selector()

