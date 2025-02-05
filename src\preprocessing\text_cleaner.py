import re
import ftfy
import cleantext
from typing import Dict, Any, List
import logging

class TextCleaner:
    """Smart text preprocessing for RAG applications."""
    
    def __init__(self):
        self.preprocessing_options = {
            'fix_unicode': True,
            'remove_headers_footers': True,
            'remove_redundant_spaces': True,
            'normalize_whitespace': True,
            'fix_line_breaks': True,
            'remove_urls': False,
            'remove_emails': False,
            'remove_phone_numbers': False,
            'remove_numbers': False,
            'remove_special_characters': False,
            'remove_multiple_spaces': True,
            'standardize_quotes': True,
            'fix_broken_unicode': True,
            'remove_empty_lines': True
        }

    def set_options(self, options: Dict[str, bool]):
        """Update preprocessing options."""
        self.preprocessing_options.update(options)

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns."""
        lines = text.split('\n')
        if len(lines) < 3:
            return text
            
        # Remove repeating lines at top and bottom
        cleaned_lines = []
        header_pattern = None
        footer_pattern = None
        
        # Detect potential headers (repeating first lines)
        for i in range(min(3, len(lines))):
            if lines[i] == lines[i + len(lines) // 2]:
                header_pattern = lines[i]
                break
        
        # Detect potential footers (repeating last lines)
        for i in range(min(3, len(lines))):
            if lines[-(i+1)] == lines[-(i+1) - len(lines) // 2]:
                footer_pattern = lines[-(i+1)]
                break
        
        # Remove detected patterns
        for line in lines:
            if (line != header_pattern and 
                line != footer_pattern and 
                not re.match(r'^Page \d+( of \d+)?$', line.strip()) and
                not re.match(r'^\d+$', line.strip())):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _fix_line_breaks(self, text: str) -> str:
        """Fix common line break issues."""
        # Fix sentence breaks
        text = re.sub(r'([.!?])\s*\n(?=[A-Z])', r'\1\n\n', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # Remove single line breaks within sentences
        text = re.sub(r'(?<=[^.!?])\n(?=[a-z])', ' ', text)
        
        return text

    def clean_text(self, text: str) -> str:
        """Apply all selected preprocessing steps."""
        if not text:
            return text

        try:
            # Fix Unicode first (using ftfy)
            if self.preprocessing_options['fix_unicode'] or self.preprocessing_options['fix_broken_unicode']:
                text = ftfy.fix_text(text)
            
            # Remove headers and footers
            if self.preprocessing_options['remove_headers_footers']:
                text = self._remove_headers_footers(text)
            
            # Basic text cleaning without cleantext library
            if self.preprocessing_options['remove_urls']:
                text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            if self.preprocessing_options['remove_emails']:
                text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
            
            if self.preprocessing_options['remove_phone_numbers']:
                text = re.sub(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)
            
            if self.preprocessing_options['remove_numbers']:
                text = re.sub(r'\d+', '', text)
            
            if self.preprocessing_options['remove_special_characters']:
                text = re.sub(r'[^\w\s\.,!?-]', '', text)
            
            # Fix line breaks
            if self.preprocessing_options['fix_line_breaks']:
                text = self._fix_line_breaks(text)
            
            # Remove multiple spaces and normalize whitespace
            if self.preprocessing_options['remove_multiple_spaces']:
                text = re.sub(r'\s+', ' ', text)
            
            if self.preprocessing_options['normalize_whitespace']:
                text = re.sub(r'[ \t]+', ' ', text)
                text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            
            if self.preprocessing_options['remove_empty_lines']:
                text = '\n'.join(line for line in text.splitlines() if line.strip())
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"Error during text cleaning: {str(e)}")
            return text  # Return original text if cleaning fails

    def get_preprocessing_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Get statistics about the preprocessing changes."""
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'reduction_percentage': round((1 - len(cleaned_text) / len(original_text)) * 100, 2),
            'removed_characters': len(original_text) - len(cleaned_text),
            'original_lines': len(original_text.splitlines()),
            'cleaned_lines': len(cleaned_text.splitlines())
        } 