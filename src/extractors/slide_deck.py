from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
from .utils.data_model import ExtractedContent, ExtractedDocument, ContentType

SLIDE_DATASET_PATH = 'dataset/owasp-top-10.pdf'

# Default texts to exclude from extraction
DEFAULT_EXCLUDED_TEXTS = [
    'Office of Information Security Securing One HHS 2',
    'Health Sector Cybersecurity Coordination Center'
]


def extract_slide_deck(file_path: str = SLIDE_DATASET_PATH) -> ExtractedDocument:
    """
    Main extraction function for slide decks.
    
    Args:
        file_path: Path to PDF slide deck
        
    Returns:
        ExtractedDocument with all contents
    """
    print(f"Extracting slide deck: {file_path}")
    
    # Extract raw elements
    elements = _extract_slide_deck_elements(file_path)
    
    # Standardize to ExtractedContent
    extracted_contents = _standardize_elements(elements, file_path)
    
    # 3. Wrap in ExtractedDocument
    doc = ExtractedDocument(
        source=file_path,
        content=extracted_contents,
        metadata={
            'total_elements': len(extracted_contents),
            'doc_type': 'slides',
            'total_slides': _count_pages(elements)
        }
    )
    
    print(f"✓ Extracted {len(extracted_contents)} items from {doc.metadata['total_slides']} slides")
    return doc


def _extract_slide_deck_elements(file_path: str) -> List[Element]:
    """
    Extract elements from a PDF slide deck using unstructured library.
    """
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
    )
    print(f'Extracted {len(elements)} elements')
    return elements


def _filter_relevant_image_elements(elements: List[Element]) -> List[Element]:
    """
    Filter out irrelevant image elements (logos, headers, footers).
    
    Args:
        elements: List of unstructured Element objects
        
    Returns:
        List of relevant image elements
    """
    relevant_images = []
    images = [el for el in elements if el.category == 'Image']
    
    for img in images:
        # Skip small images (likely logos)
        if hasattr(img.metadata, 'coordinates'):
            # You can add size filtering here if needed
            pass
        
        # Skip images with excluded text
        if img.text and img.text in DEFAULT_EXCLUDED_TEXTS:
            continue
        
        # Skip very small text in images (likely logos)
        if img.text and len(img.text) < 5:
            continue
        
        relevant_images.append(img)
    
    return relevant_images


def _standardize_elements(elements: List[Element], source_filename: str) -> List[ExtractedContent]:
    """
    Filters content and standardizes format.
    """
    clean_content_list: List[ExtractedContent] = []
    doc_id = Path(source_filename).stem
    
    # Filter relevant images first
    relevant_images = _filter_relevant_image_elements(elements)
    
    for el in elements:
        # Build base metadata
        meta = {
            "source": source_filename,
            "doc_id": doc_id,
            "page": el.metadata.page_number,
            "slide_num": el.metadata.page_number,  # For slides, page = slide
            "element_id": el.id if hasattr(el, 'id') else None
        }
        
        # Handle text elements
        if el.category in ['NarrativeText', 'Text', 'Title', 'ListItem']:
            text = el.text.strip() if el.text else ""
            if len(text) < 3:
                continue
            
            # Mark titles
            meta['is_title'] = (el.category == 'Title')
            
            clean_content_list.append(ExtractedContent(
                content_type=ContentType.TEXT,
                content=text,
                metadata=meta
            ))
        
        # Handle tables
        elif el.category == 'Table':
            table_html = None
            if hasattr(el.metadata, 'text_as_html'):
                table_html = el.metadata.text_as_html
            
            clean_content_list.append(ExtractedContent(
                content_type=ContentType.TABLE,
                content=el.text or "[Table]",
                metadata={
                    **meta,
                    "table_html": table_html
                }
            ))
    
    # Add relevant images
    for img in relevant_images:
        image_base64 = None
        if hasattr(img.metadata, 'image_base64'):
            image_base64 = img.metadata.image_base64
        
        # Create descriptive text
        description = img.text if img.text and len(img.text) > 5 else "[Slide Diagram]"
        
        clean_content_list.append(ExtractedContent(
            content_type=ContentType.DIAGRAM,
            content=description,
            metadata={
                "source": source_filename,
                "doc_id": doc_id,
                "page": img.metadata.page_number,
                "slide_num": img.metadata.page_number,
                "element_id": img.id if hasattr(img, 'id') else None
            },
            image_base64=image_base64
        ))
    
    return clean_content_list


def _count_pages(elements: List[Element]) -> int:
    """Count total slides"""
    max_page = 0
    for el in elements:
        if hasattr(el.metadata, 'page_number') and el.metadata.page_number:
            max_page = max(max_page, el.metadata.page_number)
    return max_page


if __name__ == "__main__":
    doc = extract_slide_deck()
    print(f"\nDocument: {doc.source}")
    print(f"Total elements: {len(doc.content)}")
    print(f"Text items: {len(doc.get_text_contents())}")
    print(f"Images: {len(doc.get_image_contents())}")
    print(f"Tables: {len(doc.get_table_contents())}")