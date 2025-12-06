"""Video transcription test script.

This script accepts a path to a video file and outputs the transcribed text
from the audio track.

Usage:
    python tests/manual/test_video_transcription.py <path_to_video>

Examples:
    python tests/manual/test_video_transcription.py recording.mp4
    python tests/manual/test_video_transcription.py lecture.avi
    python tests/manual/test_video_transcription.py "C:/Videos/my video.mov"
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.loaders import VideoProcessor, VideoProcessingError, UnsupportedFormatError


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def print_error(message: str):
    """Print an error message."""
    print(f"[ERROR] {message}")


def print_info(label: str, value):
    """Print an info line."""
    print(f"{label}: {value}")


def transcribe_video(video_path: str) -> str:
    """Transcribe audio from a video file."""
    print_header("VIDEO TRANSCRIPTION")
    print_info("Input file", video_path)
    
    # Check if file exists
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check file extension
    video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'}
    extension = Path(video_path).suffix.lower()
    
    if extension not in video_formats:
        print_error(f"Unsupported video format: {extension}")
        print_info("Supported formats", ", ".join(sorted(video_formats)))
        raise UnsupportedFormatError(f"Unsupported video format: {extension}")
    
    print_info("File format", extension)
    print("\nProcessing video (extracting audio and transcribing)...")
    print("-"*70)
    
    try:
        processor = VideoProcessor()
        text = processor.process_video(video_path)
        
        print("\n" + "="*70)
        print("TRANSCRIPTION COMPLETE")
        print("="*70)
        print_info("Characters", len(text))
        print_info("Words (approx)", len(text.split()))
        print_info("Lines", len(text.splitlines()))
        
        return text
        
    except (VideoProcessingError, UnsupportedFormatError) as e:
        print_error(f"Video processing failed: {e}")
        raise
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise


def main():
    """Main entry point for the video transcription script."""
    if len(sys.argv) < 2:
        print("Usage: python tests/manual/test_video_transcription.py <path_to_video>")
        print("\nExamples:")
        print("  python tests/manual/test_video_transcription.py recording.mp4")
        print("  python tests/manual/test_video_transcription.py lecture.avi")
        print('  python tests/manual/test_video_transcription.py "C:/Videos/my video.mov"')
        print("\nSupported video formats:")
        print("  .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv, .m4v, .mpg, .mpeg")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        # Transcribe the video
        transcription = transcribe_video(video_path)
        
        # Create output file path (same name as video, but with .txt extension)
        video_file = Path(video_path)
        output_path = video_file.with_suffix('.txt')
        
        # Write transcription to file
        print("\n" + "="*70)
        print("WRITING TRANSCRIPTION TO FILE")
        print("="*70)
        print_info("Output file", output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        print("\n" + "="*70)
        print("SUCCESS")
        print("="*70)
        print(f"Transcription saved to: {output_path}")
        
    except Exception as e:
        print("\n" + "="*70)
        print("TRANSCRIPTION FAILED")
        print("="*70)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
