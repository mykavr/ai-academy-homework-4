"""Manual testing script for PDFLoader.

Usage:
    python test_pdf_manual.py <path_to_pdf_file>

Example:
    python test_pdf_manual.py sample.pdf
    python test_pdf_manual.py "C:/Documents/my document.pdf"
"""

import sys
from src.loaders import PDFLoader, PDFProcessingError


def test_basic_load(pdf_path: str):
    """Test the basic load() method."""
    print("="*70)
    print("Testing load() method")
    print("="*70)
    
    loader = PDFLoader()
    
    try:
        text = loader.load(pdf_path)
        print(f"\n✓ Successfully loaded PDF: {pdf_path}")
        print(f"✓ Total characters extracted: {len(text)}")
        print(f"✓ Total lines: {len(text.splitlines())}")
        print("\n" + "-"*70)
        print("EXTRACTED TEXT:")
        print("-"*70)
        print(text)
        print("-"*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except PDFProcessingError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def test_load_with_metadata(pdf_path: str):
    """Test the load_with_metadata() method."""
    print("\n\n")
    print("="*70)
    print("Testing load_with_metadata() method")
    print("="*70)
    
    loader = PDFLoader()
    
    try:
        result = loader.load_with_metadata(pdf_path)
        
        print(f"\n✓ Successfully loaded PDF with metadata")
        print(f"✓ Source: {result['source']}")
        print(f"✓ Total pages: {result['num_pages']}")
        print(f"✓ Total characters: {len(result['text'])}")
        
        print("\n" + "-"*70)
        print("PAGE-BY-PAGE BREAKDOWN:")
        print("-"*70)
        
        for page_info in result['pages']:
            page_num = page_info['page_number']
            page_text = page_info['page_text']
            char_count = len(page_text)
            line_count = len(page_text.splitlines())
            
            print(f"\nPage {page_num}:")
            print(f"  - Characters: {char_count}")
            print(f"  - Lines: {line_count}")
            print(f"  - Preview (first 200 chars):")
            print(f"    {page_text[:200]}...")
        
        print("\n" + "-"*70)
        print("FULL TEXT:")
        print("-"*70)
        print(result['text'])
        print("-"*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except PDFProcessingError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_manual.py <path_to_pdf_file>")
        print("\nExample:")
        print("  python test_pdf_manual.py sample.pdf")
        print('  python test_pdf_manual.py "C:/Documents/my document.pdf"')
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print("\n")
    print("="*70)
    print("PDF LOADER MANUAL TEST")
    print("="*70)
    print(f"Testing with file: {pdf_path}")
    print()
    
    # Test both methods
    test_basic_load(pdf_path)
    test_load_with_metadata(pdf_path)
    
    print("\n")
    print("="*70)
    print("TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
