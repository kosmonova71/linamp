#!/usr/bin/env python3
"""
Metadata Fixer Module

This module provides functionality to fix metadata tags in audio files.
It's designed to work with the linamp_xmms application.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from mutagen import File
    from mutagen.id3 import ID3NoHeaderError
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logging.warning("mutagen library not available - limited functionality")

@dataclass
class FixResult:
    """Result of fixing metadata for a single file."""
    file_path: str
    success: bool
    message: str
    old_metadata: Optional[dict] = None
    new_metadata: Optional[dict] = None

class MetadataFixer:
    """Main class for fixing metadata in audio files."""
    
    def __init__(self):
        """Initialize the metadata fixer."""
        self.logger = logging.getLogger(__name__)
        if not MUTAGEN_AVAILABLE:
            self.logger.warning("mutagen not available - using basic filename parsing")
    
    def fix_files_threaded(self, file_paths: List[str], method: str = "filename", max_workers: int = 2) -> List[FixResult]:
        """
        Fix metadata for multiple files using threading.
        
        Args:
            file_paths: List of file paths to process
            method: Method to use for fixing metadata ("filename" or "auto")
            max_workers: Maximum number of worker threads
            
        Returns:
            List of FixResult objects
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._fix_single_file, path, method): path 
                for path in file_paths
            }
            
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    path = future_to_path[future]
                    error_result = FixResult(
                        file_path=path,
                        success=False,
                        message=f"Thread error: {str(e)}"
                    )
                    results.append(error_result)
        
        return results
    
    def _fix_single_file(self, file_path: str, method: str) -> FixResult:
        """
        Fix metadata for a single file.
        
        Args:
            file_path: Path to the audio file
            method: Method to use for fixing metadata
            
        Returns:
            FixResult object
        """
        try:
            if not os.path.exists(file_path):
                return FixResult(
                    file_path=file_path,
                    success=False,
                    message="File not found"
                )
            
            # Get old metadata
            old_metadata = self._get_metadata(file_path)
            
            if method == "filename":
                success, message = self._fix_from_filename(file_path)
            elif method == "auto":
                success, message = self._auto_fix(file_path)
            else:
                return FixResult(
                    file_path=file_path,
                    success=False,
                    message=f"Unknown method: {method}"
                )
            
            # Get new metadata if successful
            new_metadata = self._get_metadata(file_path) if success else None
            
            return FixResult(
                file_path=file_path,
                success=success,
                message=message,
                old_metadata=old_metadata,
                new_metadata=new_metadata
            )
            
        except Exception as e:
            return FixResult(
                file_path=file_path,
                success=False,
                message=f"Error: {str(e)}"
            )
    
    def _get_metadata(self, file_path: str) -> Optional[dict]:
        """Get metadata from an audio file."""
        if not MUTAGEN_AVAILABLE:
            return None
        
        try:
            audio_file = File(file_path)
            if audio_file is None:
                return None
            
            metadata = {}
            if hasattr(audio_file, 'tags') and audio_file.tags:
                for key, value in audio_file.tags.items():
                    if hasattr(value, 'text'):
                        metadata[key] = str(value.text[0]) if value.text else ""
                    else:
                        metadata[key] = str(value)
            
            return metadata
        except Exception:
            return None
    
    def _fix_from_filename(self, file_path: str) -> tuple[bool, str]:
        """
        Fix metadata by parsing the filename.
        
        Expected format: "Artist - Title.ext" or "Title.ext"
        """
        if not MUTAGEN_AVAILABLE:
            return False, "mutagen not available for metadata editing"
        
        try:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Try to parse "Artist - Title" format
            if " - " in name_without_ext:
                parts = name_without_ext.split(" - ", 1)
                artist = parts[0].strip()
                title = parts[1].strip()
            else:
                # Use filename as title
                artist = ""
                title = name_without_ext.strip()
            
            audio_file = File(file_path)
            if audio_file is None:
                return False, "Unsupported file format"
            
            # Add or update tags
            if audio_file.tags is None:
                audio_file.add_tags()
            
            # Common tag mappings
            tag_mappings = {
                'TPE1': artist,  # Artist
                'TIT2': title,   # Title
                'TITLE': title,  # Title (for other formats)
                'ARTIST': artist, # Artist (for other formats)
            }
            
            for tag, value in tag_mappings.items():
                if value:  # Only set non-empty values
                    try:
                        audio_file.tags[tag] = value
                    except KeyError:
                        pass  # Skip tags that can't be set
            
            audio_file.save()
            return True, f"Fixed metadata: {artist} - {title}"
            
        except ID3NoHeaderError:
            return False, "No ID3 header found"
        except Exception as e:
            return False, f"Error fixing metadata: {str(e)}"
    
    def _auto_fix(self, file_path: str) -> tuple[bool, str]:
        """
        Auto-fix metadata using available methods.
        Currently falls back to filename parsing.
        """
        return self._fix_from_filename(file_path)
    
    def close(self):
        """Clean up resources."""
        # Currently no specific cleanup needed
        pass

# Test function for basic functionality
def test_metadata_fixer():
    """Test the metadata fixer functionality."""
    fixer = MetadataFixer()
    
    # Test with a dummy file path
    fixer._fix_single_file("/nonexistent/file.mp3", "filename")
    
    fixer.close()

if __name__ == "__main__":
    test_metadata_fixer()
