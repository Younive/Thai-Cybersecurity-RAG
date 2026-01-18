from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from extractors.textbook import extract_textbook
from extractors.slide_deck import extract_slide_deck
from extractors.thai_pdf import extract_thai_pdf
from extractors.utils.data_model import ExtractedContent, ExtractedDocument, ContentType
from dotenv import load_dotenv
import os

EMBEDDING_MODEL = 'text-embedding-004'

load_dotenv()

class RAGPipeline:
    """
    Unified pipeline that processes ALL document types together.
    
    Extract (different extractors) → Chunk (ONE chunker) → Embed (ONE embedder) → Store (ONE DB)
    """
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str = "./chroma_db"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # ONE embedding model for ALL content
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
        
        print(f"Initialized RAG pipeline")
        print(f"Collection: {collection_name}")
        print(f"Embedding: {EMBEDDING_MODEL}")
    
    def process_all_documents(
        self,
        textbook_path: str = None,
        slides_path: str = None,
        thai_pdf_path: str = None
    ):
        """
        Process all documents through unified pipeline.
        
        Args:
            textbook_path: Path to textbook PDF
            slides_path: Path to slide deck PDF
            thai_pdf_path: Path to Thai PDF
        """
        print("\n" + "=" * 60)
        print("RAG PIPELINE - PROCESSING ALL DOCUMENTS")
        print("=" * 60)
        
        # Extract ALL documents (different extractors)
        all_extracted_contents = self._extract_all_documents(
            textbook_path, slides_path, thai_pdf_path
        )
        
        # Convert to LangChain format
        langchain_docs = self._to_langchain_docs(all_extracted_contents)
        
        # Chunk ALL content together (ONE chunker with smart logic)
        chunks = self._chunk_all_content(langchain_docs)
        
        # Store in vector database
        vectorstore = self._create_vectorstore(chunks)
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total extracted items: {len(all_extracted_contents)}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"Collection: {self.collection_name}")
        print("=" * 60)
        
        return vectorstore
    
    def _extract_all_documents(
        self,
        textbook_path: str,
        slides_path: str,
        thai_pdf_path: str
    ) -> List[ExtractedContent]:
        """
        Extract from ALL document types.
        
        Returns:
            Single list of ALL extracted contents (mixed types)
        """
        all_contents = []
        
        # Extract textbook
        if textbook_path:
            print("\nExtracting Textbook...")
            textbook_doc = extract_textbook(textbook_path)
            all_contents.extend(textbook_doc.content)
            print(f"Added {len(textbook_doc.content)} items from textbook")
        
        # Extract slides
        if slides_path:
            print("\nExtracting Slide Deck...")
            slides_doc = extract_slide_deck(slides_path)
            all_contents.extend(slides_doc.content)
            print(f"Added {len(slides_doc.content)} items from slides")
        
        # Extract Thai PDF
        if thai_pdf_path:
            print("\nExtracting Thai PDF...")
            thai_doc = extract_thai_pdf(thai_pdf_path)
            all_contents.extend(thai_doc.content)
            print(f"Added {len(thai_doc.content)} items from Thai PDF")
        
        print(f"\nTotal extracted contents: {len(all_contents)}")
        return all_contents
    
    def _to_langchain_docs(
        self,
        extracted_contents: List[ExtractedContent]
    ) -> List[Document]:
        """
        Convert ExtractedContent to LangChain Documents.
        """
        docs = []
        for item in extracted_contents:
            # Include content type in metadata
            metadata = {
                **item.metadata,
                'content_type': item.content_type.value
            }
            
            # Add table HTML if present
            if item.content_type == ContentType.TABLE and 'table_html' in item.metadata:
                metadata['table_html'] = item.metadata['table_html']
            
            # Add image flag
            if item.image_base64:
                metadata['has_image'] = True
                metadata['image_base64'] = item.image_base64
            
            docs.append(Document(
                page_content=item.content,
                metadata=metadata
            ))
        
        return docs
    
    def _chunk_all_content(self, docs: List[Document]) -> List[Document]:
        """
        Chunk ALL content with smart logic based on document type.
        ONE chunker handles everything, but with type-aware splitting.
        """
        print("\n Chunking all content...")
        
        final_chunks = []
        
        # Separate by document type for smart chunking
        textbook_docs = [d for d in docs if 'doc_type' in d.metadata and d.metadata.get('doc_type') == 'textbook']
        slide_docs = [d for d in docs if 'slide_num' in d.metadata]
        thai_docs = [d for d in docs if d.metadata.get('extraction_method') == 'pytesseract_ocr']
        other_docs = [d for d in docs if d not in textbook_docs + slide_docs + thai_docs]
        
        # Chunk textbooks (standard recursive splitting)
        if textbook_docs:
            print(f"  Chunking {len(textbook_docs)} textbook items...")
            textbook_chunks = self._chunk_textbook_style(textbook_docs)
            final_chunks.extend(textbook_chunks)
            print(f"Created {len(textbook_chunks)} textbook chunks")
        
        # Chunk slides (page-aware splitting)
        if slide_docs:
            print(f"  Chunking {len(slide_docs)} slide items...")
            slide_chunks = self._chunk_slide_style(slide_docs)
            final_chunks.extend(slide_chunks)
            print(f"Created {len(slide_chunks)} slide chunks")
        
        # Chunk Thai PDFs (smaller chunks for Thai)
        if thai_docs:
            print(f"  Chunking {len(thai_docs)} Thai PDF items...")
            thai_chunks = self._chunk_thai_style(thai_docs)
            final_chunks.extend(thai_chunks)
            print(f"Created {len(thai_chunks)} Thai chunks")
        
        # Chunk other documents
        if other_docs:
            print(f"  Chunking {len(other_docs)} other items...")
            other_chunks = self._chunk_textbook_style(other_docs)
            final_chunks.extend(other_chunks)
            print(f"Created {len(other_chunks)} other chunks")
        
        return final_chunks
    
    def _chunk_textbook_style(self, docs: List[Document]) -> List[Document]:
        """Standard recursive splitting for textbooks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(docs)
    
    def _chunk_slide_style(self, docs: List[Document]) -> List[Document]:
        """
        Page-aware splitting for slides.
        Don't mix content from different slides.
        """
        from itertools import groupby
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        final_chunks = []
        
        # Sort by slide number
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get('slide_num', 0))
        
        # Group by slide
        for slide_num, group in groupby(sorted_docs, key=lambda x: x.metadata.get('slide_num')):
            slide_items = list(group)
            
            # Separate text from tables/images
            text_items = [d for d in slide_items if d.metadata.get('content_type') == 'text']
            special_items = [d for d in slide_items if d.metadata.get('content_type') in ['table', 'diagram', 'image']]
            
            # Combine text on this slide
            if text_items:
                slide_text = "\n".join([item.page_content for item in text_items])
                slide_meta = text_items[0].metadata
                text_chunks = splitter.create_documents([slide_text], [slide_meta])
                final_chunks.extend(text_chunks)
            
            # Add tables/images as atomic chunks
            final_chunks.extend(special_items)
        
        return final_chunks
    
    def _chunk_thai_style(self, docs: List[Document]) -> List[Document]:
        """Smaller chunks for Thai text (denser information)."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(docs)
    
    def _create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Create ONE vector store for ALL chunks.
        """
        print(f"\nCreating vector store...")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        print(f"Stored {len(chunks)} chunks in {self.collection_name}")
        return vectorstore


def run_pipeline():
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        collection_name="rag_knowledge_base",
        persist_directory="./chroma_db"
    )
    
    # Process ALL documents at once
    vectorstore = pipeline.process_all_documents(
        textbook_path="dataset/mitre-attack-philosophy-2020.pdf",
        slides_path="dataset/owasp-top-10.pdf",
        thai_pdf_path="dataset/thailand-web-security-standard-2025.pdf"
    )

if __name__ == "__main__":
    run_pipeline()