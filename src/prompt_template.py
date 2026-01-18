from typing import List, Dict
from langchain_core.documents import Document

GEMINI_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base of documents.

CRITICAL CITATION RULES (YOU MUST FOLLOW THESE):
1. ALWAYS cite sources for EVERY factual claim using this format: [Source: filename, Page X]
2. NEVER make claims without citations when information comes from the retrieved documents
3. If information is NOT in the retrieved documents, explicitly say: "This information is not available in the provided documents."
4. Use EXACT page numbers from the metadata
5. Each claim needs its own citation - don't cite once at the end

CITATION BEST PRACTICES:
- If multiple facts come from the SAME page, cite once at the end of the paragraph
- Use individual citations only when facts come from DIFFERENT pages
- Don't cite every single sentence if they're all from the same source

CITATION FORMAT EXAMPLES:
CORRECT: "The OWASP Top 10 includes SQL injection vulnerabilities [Source: owasp-top-10.pdf, Page 4]."
CORRECT: "According to the MITRE ATT&CK framework, adversaries use persistence techniques [Source: mitre-attack-philosophy-2020.pdf, Page 12] to maintain access [Source: mitre-attack-philosophy-2020.pdf, Page 13]."
WRONG: "The OWASP Top 10 includes SQL injection vulnerabilities." (No citation)
WRONG: "SQL injection is a common vulnerability. [Source: owasp-top-10.pdf, Page 4]" (Citation should be right after the claim)

LANGUAGE:
- Answer in the same language as the question
- For Thai questions (คำถามภาษาไทย), answer in Thai with citations
- For English questions, answer in English with citations

REMEMBER: Every sentence with factual information MUST have a citation!"""

FEW_SHOT_EXAMPLES = """
EXAMPLE 1 - English Query:
Question: What is the first item in the OWASP Top 10?

Retrieved Documents:
[1] Source: owasp-top-10.pdf, Page: 4
Content: "A01:2021 - Broken Access Control moves up from the fifth position to the category with the most serious web application security risk."

Answer: The first item in the OWASP Top 10 (2021 edition) is A01:2021 - Broken Access Control [Source: owasp-top-10.pdf, Page 4]. This category moved up from the fifth position and represents the most serious web application security risk [Source: owasp-top-10.pdf, Page 4].

---

EXAMPLE 2 - Thai Query:
Question: มาตรฐานความปลอดภัยเว็บไซต์ของไทยมีอะไรบ้าง

Retrieved Documents:
[1] Source: thailand-web-security-standard-2025.pdf, Page: 5
Content: "มาตรฐานความปลอดภัยเว็บไซต์ภาครัฐ พ.ศ. 2568 ประกอบด้วย 5 หมวดหลัก ได้แก่ การจัดการความเสี่ยง การควบคุมการเข้าถึง การเข้ารหัสข้อมูล การตรวจสอบและบันทึก และการตอบสนองต่อเหตุการณ์"

Answer: มาตรฐานความปลอดภัยเว็บไซต์ภาครัฐ พ.ศ. 2568 ประกอบด้วย 5 หมวดหลัก [แหล่งที่มา: thailand-web-security-standard-2025.pdf, หน้า 5] ได้แก่:
1. การจัดการความเสี่ยง
2. การควบคุมการเข้าถึง
3. การเข้ารหัสข้อมูล
4. การตรวจสอบและบันทึก
5. การตอบสนองต่อเหตุการณ์
[แหล่งที่มา: thailand-web-security-standard-2025.pdf, หน้า 5]

---

EXAMPLE 3 - No Information Available:
Question: What is the current stock price of Apple?

Retrieved Documents:
[1] Source: owasp-top-10.pdf, Page: 4
Content: "A01:2021 - Broken Access Control..."

Answer: This information is not available in the provided documents. The retrieved documents contain information about web security (OWASP Top 10) but do not include stock market data.

---

Now answer the following question using the same format:"""

GEMINI_GENERATION_CONFIG = {
    "temperature": 0.5,  # Low temperature for factual accuracy
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

GEMINI_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

def build_gemini_rag_prompt(
    query: str,
    retrieved_docs: List[Document],
    language: str = "auto"
) -> str:
    """
    Build a complete RAG prompt for Gemini with citation enforcement.
    
    Args:
        query: User's question
        retrieved_docs: Retrieved documents from vector store
        language: 'en', 'th', or 'auto' to detect from query
        
    Returns:
        Complete prompt string
    """
    
    # Detect language if auto
    if language == "auto":
        thai_chars = sum(1 for c in query if '\u0E00' <= c <= '\u0E7F')
        language = 'th' if thai_chars > len(query) * 0.3 else 'en'
    
    # Format retrieved documents
    context = _format_retrieved_docs(retrieved_docs)
    
    # Build citation instruction based on language
    citation_format = _get_citation_format(language)
    
    # Build complete prompt
    prompt = f"""{GEMINI_SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLES}

Retrieved Documents:
{context}

Question: {query}

{citation_format}

Answer:"""
    
    return prompt

def _format_retrieved_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents with clear source identification.
    """
    if not docs:
        return "[No documents retrieved]"
    
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 'unknown')
        
        # Extract filename from path
        if '/' in source:
            source = source.split('/')[-1]
        
        formatted_docs.append(
            f"[{i}] Source: {source}, Page: {page}\n"
            f"Content: {doc.page_content}\n"
        )
    
    return "\n".join(formatted_docs)

def _get_citation_format(language: str) -> str:
    """Get language-specific citation instructions."""
    
    if language == 'th':
        return """คำแนะนำ:
- ตอบเป็นภาษาไทย
- อ้างอิงแหล่งที่มาทุกข้อความด้วยรูปแบบ: [แหล่งที่มา: ชื่อไฟล์, หน้า X]
- ห้ามตอบโดยไม่มีการอ้างอิง
- ถ้าไม่มีข้อมูลในเอกสาร ให้บอกว่า "ไม่มีข้อมูลนี้ในเอกสารที่ให้มา" """
    else:
        return """Instructions:
- Answer in English
- Cite EVERY claim using format: [Source: filename, Page X]
- Never answer without citations
- If information is not in documents, say "This information is not available in the provided documents." """

def extract_citations(response: str) -> List[Dict[str, str]]:
    """
    Extract citations from the model's response.
    
    Args:
        response: Model's response text
        
    Returns:
        List of dictionaries containing source and page information
    """
    import re
    
    # Pattern for English citations: [Source: filename, Page X]
    en_pattern = r'\[Source:\s*([^,]+),\s*Page\s*(\d+|unknown)\]'
    
    # Pattern for Thai citations: [แหล่งที่มา: filename, หน้า X]
    th_pattern = r'\[แหล่งที่มา:\s*([^,]+),\s*หน้า\s*(\d+|unknown)\]'
    
    citations = []
    
    # Extract English citations
    for match in re.finditer(en_pattern, response):
        citations.append({
            'source': match.group(1).strip(),
            'page': match.group(2).strip()
        })
    
    # Extract Thai citations
    for match in re.finditer(th_pattern, response):
        citations.append({
            'source': match.group(1).strip(),
            'page': match.group(2).strip()
        })
    
    return citations