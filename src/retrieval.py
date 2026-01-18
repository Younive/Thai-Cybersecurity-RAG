from vectorstore.manage_vectorstore import VectorStoreManager
from prompt_template import build_gemini_rag_prompt
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

# Initialize vectorstore once
vectorstore = VectorStoreManager().get_exist_cromadb()

# Multilingual keyword mappings for query expansion
THAI_SECURITY_KEYWORDS = {
    # Web security terms
    "web security": "ความปลอดภัยเว็บไซต์",
    "website security": "ความปลอดภัยเว็บไซต์",
    "security standard": "มาตรฐานความปลอดภัย",
    "security controls": "มาตรการความปลอดภัย",
    "controls": "มาตรการ",
    "government": "ภาครัฐ",
    "requirements": "ข้อกำหนด",
    
    # Common security concepts
    "access control": "การควบคุมการเข้าถึง",
    "authentication": "การยืนยันตัวตน",
    "encryption": "การเข้ารหัส",
    "monitoring": "การตรวจสอบ",
    "incident response": "การตอบสนองเหตุการณ์",
    "risk management": "การจัดการความเสี่ยง",
    "logging": "การบันทึก",
    "audit": "การตรวจสอบ",
    
    # OWASP terms
    "vulnerability": "ช่องโหว่",
    "injection": "การแทรก",
    "broken access": "การควบคุมการเข้าถึงที่เสียหาย",
}

ENGLISH_SECURITY_KEYWORDS = {v: k for k, v in THAI_SECURITY_KEYWORDS.items()}


def detect_language(text: str) -> str:
    """
    Detect if text is Thai or English.
    
    Args:
        text: Input text
        
    Returns:
        'th' for Thai, 'en' for English
    """
    # Count Thai characters
    thai_chars = sum(1 for c in text if '\u0E00' <= c <= '\u0E7F')
    total_chars = len([c for c in text if c.isalnum()])
    
    if total_chars == 0:
        return 'en'
    
    thai_ratio = thai_chars / total_chars if total_chars > 0 else 0
    return 'th' if thai_ratio > 0.3 else 'en'


def expand_query_with_translations(query: str, detected_lang: str) -> list:
    """
    Expand query with bilingual keywords.
    
    Args:
        query: Original query
        detected_lang: Detected language ('en' or 'th')
        
    Returns:
        List of expanded queries
    """
    queries = [query]  # include original
    
    if detected_lang == 'en':
        # English query - add Thai translations
        expanded_query = query
        for en_term, th_term in THAI_SECURITY_KEYWORDS.items():
            if en_term.lower() in query.lower():
                expanded_query = expanded_query + " " + th_term
        
        if expanded_query != query:
            queries.append(expanded_query)
            
        # create a pure Thai query for better matching
        thai_only = []
        for en_term, th_term in THAI_SECURITY_KEYWORDS.items():
            if en_term.lower() in query.lower():
                thai_only.append(th_term)
        
        if thai_only:
            queries.append(" ".join(thai_only))
    
    elif detected_lang == 'th':
        # Thai query - add English translations
        expanded_query = query
        for th_term, en_term in ENGLISH_SECURITY_KEYWORDS.items():
            if th_term in query:
                expanded_query = expanded_query + " " + en_term
        
        if expanded_query != query:
            queries.append(expanded_query)
    
    return queries


def is_thai_related_query(query: str) -> bool:
    """
    Check if query is related to Thai content.
    
    Args:
        query: Search query
        
    Returns:
        True if likely asking about Thai content
    """
    thai_indicators = [
        'thailand', 'thai', 'ไทย', 'ภาครัฐ', 'มาตรฐาน',
        'พ.ศ.', 'ncsa', 'government'
    ]
    
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in thai_indicators)


def filter_irrelevant_pages(results_with_scores, query: str) -> list:
    """
    Filter out likely irrelevant pages (bibliography, TOC, etc.).
    
    Args:
        results_with_scores: List of (doc, score) tuples
        query: Original query
        
    Returns:
        Filtered list of (doc, score) tuples
    """
    irrelevant_indicators = [
        'บรรณานุกรม',  # Bibliography in Thai
        'bibliography',
        'references',
        'สารบัญ',  # Table of contents in Thai
        'table of contents',
        'อ้างอิง',  # References in Thai
    ]
    
    filtered = []
    for doc, score in results_with_scores:
        content_lower = doc.page_content.lower()
        
        # Check if this is a bibliography/reference page
        is_irrelevant = any(
            indicator in content_lower 
            for indicator in irrelevant_indicators
        )
        
        # Keep if not irrelevant OR if we have very few results
        if not is_irrelevant:
            filtered.append((doc, score))
    
    # If filtering removed everything, return original (with warning)
    if not filtered and results_with_scores:
        print("⚠️  Warning: All results filtered as irrelevant, using original results")
        return results_with_scores
    
    return filtered


def retrieve_documents(query: str, k: int = 3):
    """
    Retrieve relevant documents from the vector store.
    
    Args:
        query: Search query string
        k: Number of documents to retrieve
        
    Returns:
        List of Document objects with metadata
    """
    results = vectorstore.similarity_search(query, k=k)
    return results


def retrieve_documents_multilingual(query: str, k: int = 5, adaptive_k: bool = True, filter_pages: bool = True):
    """
    Retrieve documents with multilingual query expansion.
    
    Strategy:
    1. Detect query language
    2. Search with original query
    3. Expand with translations and search again
    4. Filter irrelevant pages (bibliography, TOC)
    5. Deduplicate and rank by relevance
    
    Args:
        query: Search query string
        k: Number of documents to retrieve
        adaptive_k: If True, increase k for Thai queries
        filter_pages: If True, filter out bibliography/TOC pages
        
    Returns:
        List of unique Document objects ranked by relevance
    """
    detected_lang = detect_language(query)
    original_k = k  # Store original k value
    
    # Adaptive k: Thai queries need more docs due to embedding challenges
    if adaptive_k and (detected_lang == 'th' or is_thai_related_query(query)):
        k = min(k + 5, 15)  # Increase by 5, max 15
        print(f"Thai-related query detected. Increasing k to {k}")
    
    # Try original query first
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # Always expand for better coverage
    print(f"Expanding query with translations...")
    expanded_queries = expand_query_with_translations(query, detected_lang)
    
    # Collect all results from expanded queries
    for expanded_query in expanded_queries[1:]:  # Skip original
        if expanded_query.strip() != query.strip():  # Avoid duplicate queries
            try:
                additional_results = vectorstore.similarity_search_with_score(
                    expanded_query, 
                    k=k  # Get full k for each expanded query
                )
                results.extend(additional_results)
                print(f"   Added {len(additional_results)} results from expanded query")
            except Exception as e:
                print(f"Warning: Expanded query failed: {e}")
                continue
    
    # Filter out irrelevant pages
    if filter_pages:
        results = filter_irrelevant_pages(results, query)
        print(f"After filtering: {len(results)} results")
    
    # Deduplicate based on content
    seen_content = set()
    unique_results = []
    
    for doc, score in results:
        # Use first 100 chars as fingerprint
        fingerprint = doc.page_content[:100]
        if fingerprint not in seen_content:
            seen_content.add(fingerprint)
            unique_results.append((doc, score))
    
    # Sort by score (lower is better in ChromaDB)
    unique_results.sort(key=lambda x: x[1])
    
    # Return top k documents only (use original_k to match user's request)
    return [doc for doc, score in unique_results[:original_k]]


def retrieve_with_scores(query: str, k: int = 3):
    """
    Retrieve relevant documents with similarity scores.
    
    Args:
        query: Search query string
        k: Number of documents to retrieve
        
    Returns:
        List of tuples (Document, score)
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def retrieve_hybrid(query: str, k: int = 5, use_multilingual: bool = True):
    """
    Hybrid retrieval: Semantic + Multilingual expansion.
    
    Args:
        query: Search query
        k: Number of results
        use_multilingual: Enable multilingual expansion
        
    Returns:
        List of documents
    """
    if use_multilingual:
        return retrieve_documents_multilingual(query, k=k)
    else:
        return retrieve_documents(query, k=k)


def get_rag_prompt(query: str, k: int = 3, language: str = "auto", use_multilingual: bool = True):
    """
    Get complete RAG prompt ready for LLM inference.
    
    Args:
        query: User's question
        k: Number of documents to retrieve
        language: 'en', 'th', or 'auto'
        use_multilingual: Enable multilingual retrieval
        
    Returns:
        Tuple of (prompt string, retrieved documents)
    """
    if use_multilingual:
        results = retrieve_documents_multilingual(query, k=k)
    else:
        results = retrieve_documents(query, k=k)
    
    prompt = build_gemini_rag_prompt(query, results, language=language)
    return prompt, results


def main():
    """Test retrieval functionality with multilingual examples"""
    print("\n" + "="*80)
    print("Testing Multilingual Retrieval System")
    print("="*80)
    
    # Test 1: English query about Thai content (THE PROBLEMATIC ONE)
    print("\nTEST 1: English Query about Thai Content")
    print("-"*80)
    query_en = "What website security controls are required by the Thailand Web Security Standard?"
    print(f"Query: {query_en}")
    
    results_en = retrieve_documents_multilingual(query_en, k=8)  # Increased k
    print(f"\nFound {len(results_en)} relevant chunks:")
    for i, doc in enumerate(results_en, 1):
        print(f"{i}. Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Page: {doc.metadata.get('page', 'unknown')}")
        print(f"   Preview: {doc.page_content[:150]}...")
        print()
    
    # Test 2: Thai query
    print("\n" + "="*80)
    print("TEST 2: Thai Query")
    print("-"*80)
    query_th = "มาตรฐานความปลอดภัยเว็บไซต์ของไทยมีอะไรบ้าง"
    print(f"Query: {query_th}")
    
    results_th = retrieve_documents_multilingual(query_th, k=8)
    print(f"\nFound {len(results_th)} relevant chunks:")
    for i, doc in enumerate(results_th, 1):
        print(f"{i}. Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Page: {doc.metadata.get('page', 'unknown')}")
        print(f"   Preview: {doc.page_content[:150]}...")
        print()
    
    # Test 3: Full RAG with the problematic query
    print("\n" + "="*80)
    print("TEST 3: Full RAG Pipeline (English Query for Thai Content)")
    print("="*80)
    test_query = 'How does MITRE describe the purpose of Persistence techniques?'
    prompt, docs = get_rag_prompt(test_query, k=8, language="auto", use_multilingual=True)
    
    model = GoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    response = model.invoke(prompt)
    
    print(f"\nQuery: {test_query}")
    print(f"Retrieved: {len(docs)} documents")
    print("\nRAG Response:")
    print("-"*80)
    print(response)
    print("-"*80)


if __name__ == "__main__":
    main()