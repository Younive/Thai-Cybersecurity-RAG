from pathlib import Path
from typing import List, Dict
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
from pdf2image import convert_from_path
import pytesseract
from .utils.data_model import ExtractedContent, ExtractedDocument, ContentType

THAI_PDF_DATASET_PATH = "dataset/thailand-web-security-standard-2025.pdf"


def extract_thai_pdf(file_path: str = THAI_PDF_DATASET_PATH) -> ExtractedDocument:
    """
    Main extraction function for Thai PDFs.
    Combines OCR text with extracted images/tables.
    
    Args:
        file_path: Path to Thai PDF
        
    Returns:
        ExtractedDocument with all contents
    """
    print(f"Extracting Thai PDF: {file_path}")
    
    # Extract images and tables with Unstructured
    visual_elements = _extract_visual_elements(Path(file_path))
    
    # OCR the entire document for text
    ocr_text_by_page = _ocr_document(Path(file_path))
    
    # Combine and standardize
    extracted_contents = _standardize_elements(
        ocr_text_by_page, 
        visual_elements, 
        file_path
    )
    
    # Wrap in ExtractedDocument
    doc = ExtractedDocument(
        source=file_path,
        content=extracted_contents,
        metadata={
            'total_elements': len(extracted_contents),
            'doc_type': 'thai_pdf',
            'total_pages': len(ocr_text_by_page),
            'extraction_method': 'pytesseract_ocr + unstructured'
        }
    )
    
    print(f"Extracted {len(extracted_contents)} items from {doc.metadata['total_pages']} pages")
    return doc


def _extract_visual_elements(file_path: Path) -> List[Element]:
    """
    Extract images and tables using Unstructured.
    """
    print(f"Extracting images and tables from: {file_path}")
    elements = partition_pdf(
        filename=str(file_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        languages=["tha+eng"]
    )
    
    # Filter only images and tables
    visual_elements = [el for el in elements if el.category in ['Image', 'Table']]
    print(f"Found {len(visual_elements)} visual elements")
    return visual_elements


def _ocr_document(file_path: Path) -> List[Dict[str, any]]:
    """
    OCR the entire document page by page.
    
    Returns:
        List of dicts with 'page' and 'text' keys
    """
    print(f"OCR processing: {file_path}")
    images = convert_from_path(str(file_path), dpi=300)
    
    ocr_results = []
    for i, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(
            img,
            lang="tha+eng",
            config="--psm 6 --oem 1"
        )
        
        # Clean the text
        text = _clean_thai_text(text)
        
        if text.strip():  # Only add non-empty pages
            ocr_results.append({
                'page': i,
                'text': text
            })
    
    print(f"OCR completed: {len(ocr_results)} pages")
    return ocr_results


def _clean_thai_text(text: str) -> str:
    """
    Clean OCR artifacts from Thai text.
    """
    # Remove excessive whitespace
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove common OCR artifacts
    text = text.replace('|', 'I')  # Common OCR mistake
    
    # Normalize Thai characters if pythainlp is available
    try:
        from pythainlp.util import normalize
        text = normalize(text)
    except ImportError:
        pass
    
    return text.strip()


def _standardize_elements(
    ocr_text_by_page: List[Dict],
    visual_elements: List[Element],
    filename: str
) -> List[ExtractedContent]:
    """
    Merges OCR text and visual elements into chronological order.
    """
    combined_content: List[ExtractedContent] = []
    doc_id = Path(filename).stem
    
    #  Add OCR text (one ExtractedContent per page)
    for item in ocr_text_by_page:
        combined_content.append(ExtractedContent(
            content_type=ContentType.TEXT,
            content=item['text'],
            metadata={
                "source": filename,
                "doc_id": doc_id,
                "page": item['page'],
                "extraction_method": "pytesseract_ocr"
            }
        ))
    
    # Add visual elements (tables and images)
    for el in visual_elements:
        base_meta = {
            "source": filename,
            "doc_id": doc_id,
            "page": el.metadata.page_number,
            "extraction_method": "unstructured_hires",
            "element_id": el.id if hasattr(el, 'id') else None
        }
        
        if el.category == 'Table':
            table_html = None
            if hasattr(el.metadata, 'text_as_html'):
                table_html = el.metadata.text_as_html
            
            combined_content.append(ExtractedContent(
                content_type=ContentType.TABLE,
                content=el.text or "[ตาราง]",  # Thai for "Table"
                metadata={
                    **base_meta,
                    "table_html": table_html
                }
            ))
        
        elif el.category == 'Image':
            image_base64 = None
            if hasattr(el.metadata, 'image_base64'):
                image_base64 = el.metadata.image_base64
            
            combined_content.append(ExtractedContent(
                content_type=ContentType.DIAGRAM,
                content="[แผนภาพ/รูปภาพ]",  # Thai for "Diagram/Image"
                metadata=base_meta,
                image_base64=image_base64
            ))
    
    # Sort by page number
    combined_content.sort(key=lambda x: x.metadata['page'])
    
    return combined_content


if __name__ == "__main__":
    doc = extract_thai_pdf()
    print(f"\nDocument: {doc.source}")
    print(f"Total elements: {len(doc.content)}")
    print(f"Text items: {len(doc.get_text_contents())}")
    print(f"Images: {len(doc.get_image_contents())}")
    print(f"Tables: {len(doc.get_table_contents())}")