from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
from .utils.data_model import ExtractedContent, ExtractedDocument, ContentType

TEXTBOOK_DATASET_PATH = 'dataset/mitre-attack-philosophy-2020.pdf'


def extract_textbook(file_path: str = TEXTBOOK_DATASET_PATH) -> ExtractedDocument:
    """
    Main extraction function that returns a complete ExtractedDocument.
    
    Args:
        file_path: Path to PDF textbook
        
    Returns:
        ExtractedDocument with all contents
    """
    print(f"Extracting textbook: {file_path}")
    
    # Extract raw elements
    elements = _extract_textbook_elements(file_path)
    
    # Standardize to ExtractedContent
    extracted_contents = _standardize_textbook_content(elements, file_path)
    
    # Wrap in ExtractedDocument
    doc = ExtractedDocument(
        source=file_path,
        content=extracted_contents,
        metadata={
            'total_elements': len(extracted_contents),
            'doc_type': 'textbook',
            'total_pages': _count_pages(elements)
        }
    )
    
    print(f"Extracted {len(extracted_contents)} items from {doc.metadata['total_pages']} pages")
    return doc


def _extract_textbook_elements(file_path: str) -> List[Element]:
    """
    Extracts elements using hi_res to capture diagrams and complex layouts.
    """
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True
    )
    print(f"Extracted {len(elements)} raw elements")
    return elements


def _is_header_or_footer(element: Element) -> bool:
    """
    Heuristic to filter out Page Numbers and Running Headers.
    """
    if element.category in ['Header', 'Footer']:
        return True
    
    # Check for standalone numbers (Page numbers)
    if element.text and element.text.strip().isdigit():
        return True
    
    return False


def _standardize_textbook_content(
    elements: List[Element], 
    filename: str
) -> List[ExtractedContent]:
    """
    Convert Unstructured elements to ExtractedContent objects.
    """
    extracted_content = []
    doc_id = Path(filename).stem
    
    for el in elements:
        # Filter Noise
        if _is_header_or_footer(el):
            continue
        
        # Skip if no text and no image
        if not el.text and not hasattr(el.metadata, 'image_base64'):
            continue
        
        # Common Metadata
        metadata = {
            "source": filename,
            "doc_id": doc_id,
            "page": el.metadata.page_number,
            "original_category": el.category,
            "element_id": el.id if hasattr(el, 'id') else None
        }
        
        # Handle Different Types
        if el.category == 'Table':
            table_html = None
            if hasattr(el.metadata, 'text_as_html'):
                table_html = el.metadata.text_as_html
            
            extracted_content.append(ExtractedContent(
                content_type=ContentType.TABLE,
                content=el.text or "[Table]",
                metadata={
                    **metadata,
                    "table_html": table_html
                }
            ))
        
        elif el.category == 'Image':
            # Textbooks have diagrams
            description = el.text if el.text and len(el.text) > 5 else "[Diagram/Figure]"
            
            image_base64 = None
            if hasattr(el.metadata, 'image_base64'):
                image_base64 = el.metadata.image_base64
            
            extracted_content.append(ExtractedContent(
                content_type=ContentType.DIAGRAM,
                content=description,
                metadata=metadata,
                image_base64=image_base64
            ))
        
        # Group Titles, NarrativeText, and ListItems as TEXT
        elif el.category in ['Title', 'NarrativeText', 'ListItem', 'Text', 'UncategorizedText']:
            # Skip empty text
            if not el.text or not el.text.strip():
                continue
            
            # Mark if it's a title (useful for chunking)
            metadata['is_title'] = (el.category == 'Title')
            
            extracted_content.append(ExtractedContent(
                content_type=ContentType.TEXT,
                content=el.text.strip(),
                metadata=metadata
            ))
    
    return extracted_content


def _count_pages(elements: List[Element]) -> int:
    """Count total pages from elements"""
    max_page = 0
    for el in elements:
        if hasattr(el.metadata, 'page_number') and el.metadata.page_number:
            max_page = max(max_page, el.metadata.page_number)
    return max_page


if __name__ == "__main__":
    doc = extract_textbook()
    print(f"\nDocument: {doc.source}")
    print(f"Total elements: {len(doc.content)}")
    print(f"Text items: {len(doc.get_text_contents())}")
    print(f"Images: {len(doc.get_image_contents())}")
    print(f"Tables: {len(doc.get_table_contents())}")