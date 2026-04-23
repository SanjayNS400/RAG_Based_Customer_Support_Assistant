"""
document_processor.py
---------------------
Module 1: Document Processing
Loads PDF files and returns LangChain Document objects.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


class DocumentProcessor:
    """
    Loads a PDF from disk using LangChain's PyPDFLoader.
    Each page becomes a separate Document object with page metadata.
    """

    def validate_file(self, path: str) -> None:
        """Validate that the file exists and is a PDF."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"PDF file not found: '{path}'\n"
                f"Please check the path and try again."
            )
        if not path.lower().endswith(".pdf"):
            raise ValueError(
                f"Expected a .pdf file, got: '{path}'\n"
                f"Only PDF files are supported."
            )
        if os.path.getsize(path) == 0:
            raise ValueError(f"The file '{path}' is empty.")

    def load_pdf(self, path: str) -> List[Document]:
        """
        Load a PDF and return a list of Document objects (one per page).

        Args:
            path: Absolute or relative path to the PDF file.

        Returns:
            List of LangChain Document objects with page_content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If PDF parsing fails.
        """
        self.validate_file(path)

        try:
            loader = PyPDFLoader(path)
            documents = loader.load()
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse PDF '{path}': {e}\n"
                f"The file may be corrupted, password-protected, or in an unsupported format."
            )

        if not documents:
            raise ValueError(
                f"No text could be extracted from '{path}'.\n"
                f"The PDF may contain only scanned images (no text layer)."
            )

        # Enrich metadata
        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(path)

        print(f"[DocumentProcessor] Loaded {len(documents)} pages from '{os.path.basename(path)}'")
        return documents
