"""
llm_handler.py
--------------
Module 6: Query Processing / LLM Interaction
Builds prompts and calls the LLM to generate grounded answers.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import config


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful and professional customer support assistant.

Your job is to answer the user's question accurately using ONLY the information 
provided in the CONTEXT below. Do not use any outside knowledge.

Rules:
1. If the context contains the answer, give a clear, concise, and friendly response.
2. If the context does NOT contain enough information, respond with exactly:
   "I cannot find sufficient information in our knowledge base to answer this question."
3. Always be polite and professional.
4. Cite which part of the context your answer comes from when possible.
5. Do not make up information, prices, dates, or policies not found in the context.

CONTEXT:
{context}
"""

HUMAN_PROMPT = "Customer Question: {question}"

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])


def get_llm():
    """
    Factory: returns the appropriate LangChain LLM object.
    Uses OpenAI GPT if API key is available, else falls back to local Ollama.
    """
    if config.USE_OPENAI:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=config.OPENAI_LLM_MODEL,
            temperature=0,                   # deterministic answers
            openai_api_key=config.OPENAI_API_KEY,
        )
        print(f"[LLMHandler] Using OpenAI LLM: {config.OPENAI_LLM_MODEL}")
        return llm
    else:
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=config.OLLAMA_LLM_MODEL,
            temperature=0,
        )
        print(f"[LLMHandler] Using local Ollama LLM: {config.OLLAMA_LLM_MODEL}")
        return llm


class LLMHandler:
    """
    Handles LLM interaction: prompt construction, API call, and response parsing.
    """

    def __init__(self):
        self.llm = get_llm()
        self.chain = PROMPT_TEMPLATE | self.llm | StrOutputParser()

    def format_context(self, chunks: List[Document]) -> str:
        """
        Format retrieved chunks into a single context string for the prompt.
        Each chunk is labelled with its source and page number.
        """
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source_file", "document")
            page = chunk.metadata.get("page", "?")
            parts.append(
                f"[Source {i}: {source}, Page {page + 1}]\n{chunk.page_content}"
            )

        return "\n\n---\n\n".join(parts)

    def generate_answer(self, query: str, chunks: List[Document]) -> str:
        """
        Generate an answer using the LLM with retrieved chunks as context.

        Args:
            query:  The user's question.
            chunks: Retrieved Document chunks from ChromaDB.

        Returns:
            The LLM's generated answer string.

        Raises:
            RuntimeError: If the LLM API call fails.
        """
        context = self.format_context(chunks)

        try:
            answer = self.chain.invoke({
                "context": context,
                "question": query,
            })
            return answer.strip()
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")
