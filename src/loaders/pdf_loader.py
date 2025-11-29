"""PDF loader module for extracting text from PDF files."""

import pdfplumber
from pathlib import Path
from typing import Dict, List


class PDFProcessingError(Exception):
    """Exception raised when PDF processing fails."""
    pass


class PDFLoader:
    """Loads and extracts text from PDF files using pdfplumber."""
    
    def load(self, file_path: str) -> str:
        """
        Extract all text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            PDFProcessingError: If the PDF is corrupted or cannot be processed
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Check if file is a PDF
        if path.suffix.lower() != '.pdf':
            raise PDFProcessingError(f"File is not a PDF: {file_path}")
        
        try:
            extracted_text = []
            
            with pdfplumber.open(file_path) as pdf:
                # Check if PDF has pages
                if len(pdf.pages) == 0:
                    raise PDFProcessingError(f"PDF has no pages: {file_path}")
                
                # Extract text from each page in order
                for page in pdf.pages:
                    page_text = page.extract_text()
                    
                    # Handle pages with no extractable text
                    if page_text:
                        extracted_text.append(page_text)
            
            # Join all pages with newlines to preserve order
            full_text = "\n".join(extracted_text)
            
            return full_text
            
        except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as e:
            raise PDFProcessingError(
                f"Corrupted or invalid PDF file: {file_path}. Error: {str(e)}"
            )
        except Exception as e:
            raise PDFProcessingError(
                f"Failed to process PDF: {file_path}. Error: {str(e)}"
            )
    
    def load_with_metadata(self, file_path: str) -> Dict[str, any]:
        """
        Extract text from a PDF file with metadata including page numbers.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - 'text': Full extracted text
                - 'pages': List of dicts with page_number and page_text
                - 'num_pages': Total number of pages
                - 'source': Source file path
                
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            PDFProcessingError: If the PDF is corrupted or cannot be processed
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Check if file is a PDF
        if path.suffix.lower() != '.pdf':
            raise PDFProcessingError(f"File is not a PDF: {file_path}")
        
        try:
            pages_data = []
            extracted_text = []
            
            with pdfplumber.open(file_path) as pdf:
                # Check if PDF has pages
                if len(pdf.pages) == 0:
                    raise PDFProcessingError(f"PDF has no pages: {file_path}")
                
                # Extract text from each page with metadata
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    
                    # Handle pages with no extractable text
                    if page_text:
                        pages_data.append({
                            'page_number': page_num,
                            'page_text': page_text
                        })
                        extracted_text.append(page_text)
            
            # Combine all text
            full_text = "\n".join(extracted_text)
            
            return {
                'text': full_text,
                'pages': pages_data,
                'num_pages': len(pages_data),
                'source': str(path)
            }
            
        except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as e:
            raise PDFProcessingError(
                f"Corrupted or invalid PDF file: {file_path}. Error: {str(e)}"
            )
        except Exception as e:
            raise PDFProcessingError(
                f"Failed to process PDF: {file_path}. Error: {str(e)}"
            )
