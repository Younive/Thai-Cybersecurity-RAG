"""
data_model.py
Fixed data model with proper structure
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum


class ContentType(Enum):
    """Types of content that can be extracted"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    DIAGRAM = "diagram"


@dataclass
class ExtractedContent:
    """Structured container for extracted content"""
    content: str  # Main text content or description
    content_type: ContentType
    metadata: Dict[str, Any]
    image_path: Optional[Path] = None
    image_base64: Optional[str] = None
    
    def __post_init__(self):
        """Backfill page/doc_id metadata when the extractor didn't set them."""
        if 'page' not in self.metadata:
            self.metadata['page'] = self.metadata.get('page_number', 0)
        if 'doc_id' not in self.metadata:
            source = self.metadata.get('source', 'unknown')
            self.metadata['doc_id'] = Path(source).stem if source != 'unknown' else 'unknown'


@dataclass
class ExtractedDocument:
    """Container for a fully extracted document"""
    source: str
    content: List[ExtractedContent]
    metadata: Dict[str, Any]
    
    @property
    def doc_id(self) -> str:
        """Get document ID from source"""
        return Path(self.source).stem
    
    def get_text_contents(self) -> List[ExtractedContent]:
        """Get only text contents"""
        return [c for c in self.content if c.content_type == ContentType.TEXT]
    
    def get_image_contents(self) -> List[ExtractedContent]:
        """Get only image/diagram contents"""
        return [c for c in self.content 
                if c.content_type in [ContentType.IMAGE, ContentType.DIAGRAM]]
    
    def get_table_contents(self) -> List[ExtractedContent]:
        """Get only table contents"""
        return [c for c in self.content if c.content_type == ContentType.TABLE]