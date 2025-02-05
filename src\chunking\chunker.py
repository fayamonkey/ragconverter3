from typing import List, Dict, Any
import re
import spacy
from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Split text into chunks according to the strategy."""
        pass

class LengthBasedChunker(ChunkingStrategy):
    """Chunks text based on character length."""
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position with overlap
            end = start + chunk_size
            
            # If not at the end, try to break at a sentence or paragraph
            if end < text_length:
                # Look for paragraph break
                next_para = text.find('\n\n', end - 50, end + 50)
                if next_para != -1 and next_para - end < 50:
                    end = next_para
                else:
                    # Look for sentence break
                    next_period = text.find('.', end - 30, end + 30)
                    if next_period != -1:
                        end = next_period + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start': start,
                    'end': end,
                    'strategy': 'length_based'
                })
            
            # Calculate next start position with overlap
            start = end - overlap
        
        return chunks

class SemanticChunker(ChunkingStrategy):
    """Chunks text based on semantic understanding using spaCy."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If English model is not available, try loading it
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
            except:
                raise RuntimeError("No spaCy model available. Please install with: python -m spacy download en_core_web_sm")
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        chunks = []
        doc = self.nlp(text)
        
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)
            
            if current_length + sent_length > chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start': text.find(current_chunk[0]),
                    'end': text.find(current_chunk[-1]) + len(current_chunk[-1]),
                    'strategy': 'semantic'
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
            
            current_chunk.append(sent_text)
            current_length += sent_length
        
        # Add remaining sentences as last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start': text.find(current_chunk[0]),
                'end': text.find(current_chunk[-1]) + len(current_chunk[-1]),
                'strategy': 'semantic'
            })
        
        return chunks

class StructureBasedChunker(ChunkingStrategy):
    """Chunks text based on document structure (headers, paragraphs, etc.)."""
    
    def __init__(self):
        self.header_pattern = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        chunks = []
        
        # Split by headers for markdown
        if text.startswith('#'):
            sections = self.header_pattern.split(text)
            headers = self.header_pattern.findall(text)
            
            for i, (header, content) in enumerate(zip(['']+headers, sections)):
                if not content.strip():
                    continue
                
                chunk_text = (header + '\n' + content).strip()
                chunks.append({
                    'text': chunk_text,
                    'start': text.find(chunk_text),
                    'end': text.find(chunk_text) + len(chunk_text),
                    'strategy': 'structure_based',
                    'level': len(header.split('#')[0]) if header else 0
                })
        else:
            # Split by paragraphs for plain text
            paragraphs = text.split('\n\n')
            start = 0
            
            for para in paragraphs:
                if not para.strip():
                    continue
                
                end = start + len(para)
                chunks.append({
                    'text': para.strip(),
                    'start': start,
                    'end': end,
                    'strategy': 'structure_based',
                    'level': 0
                })
                start = end + 2  # Account for '\n\n'
        
        return chunks

class Chunker:
    """Main chunking class that coordinates different chunking strategies."""
    
    def __init__(self):
        self.strategies = {
            'Length-based': LengthBasedChunker(),
            'Semantic': SemanticChunker(),
            'Structure-based': StructureBasedChunker()
        }
    
    def chunk_document(self, text: str, strategy: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        Chunk a document using the specified strategy.
        
        Args:
            text: The text to chunk
            strategy: The chunking strategy to use
            chunk_size: The target size of each chunk
            overlap: The number of characters to overlap between chunks
            
        Returns:
            List of chunks with metadata
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        chunker = self.strategies[strategy]
        return chunker.chunk_text(text, chunk_size, overlap) 