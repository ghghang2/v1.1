#!/usr/bin/env python3
"""
Standalone V4A Diff Patch Tool

This module implements the V4A (Version 4A) diff format used by OpenAI's GPT models
for code editing. It provides a complete standalone implementation of apply_diff
functionality without requiring the openai-agents SDK.

V4A Format Specification:
    - Headerless diff format (no file headers like traditional unified diff)
    - Hunk headers: @@ [optional context description]
    - Line prefixes:
        '+' = added line
        '-' = removed line
        ' ' (space) = context line (unchanged)
    - Hunks are applied sequentially to the file content

Usage:
    # Create a new file from diff
    content = apply_diff("", diff_text, mode="create")
    
    # Update an existing file
    new_content = apply_diff(current_content, diff_text, mode=None)
    
    # Command line
    python apply_patch.py --file input.txt --diff patch.diff --output output.txt
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional


class DiffApplicationError(Exception):
    """Raised when a diff cannot be applied to the content."""
    pass


@dataclass
class DiffHunk:
    """Represents a single hunk in a V4A diff."""
    context_description: Optional[str]
    lines: List[tuple[str, str]]  # (prefix, content) pairs
    
    def __repr__(self) -> str:
        desc = f" {self.context_description}" if self.context_description else ""
        return f"<DiffHunk{desc} with {len(self.lines)} lines>"


def parse_v4a_diff(diff_text: str) -> List[DiffHunk]:
    """
    Parse a V4A diff into structured hunks.
    
    Args:
        diff_text: The V4A diff string
        
    Returns:
        List of DiffHunk objects
        
    Raises:
        DiffApplicationError: If the diff format is invalid
    """
    hunks: List[DiffHunk] = []
    current_hunk: Optional[DiffHunk] = None
    
    lines = diff_text.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        # Hunk header: @@ [optional description]
        if line.startswith('@@'):
            # Save previous hunk if it exists
            if current_hunk is not None:
                hunks.append(current_hunk)
            
            # Extract optional context description
            context_match = re.match(r'^@@\s*(.*?)\s*$', line)
            context_description = context_match.group(1) if context_match and context_match.group(1) else None
            current_hunk = DiffHunk(context_description=context_description, lines=[])
            
        elif line.startswith(('+', '-', ' ')):
            if current_hunk is None:
                raise DiffApplicationError(
                    f"Line {line_num}: Found diff line before hunk header: {line[:50]}"
                )
            
            prefix = line[0]
            content = line[1:]  # Everything after the prefix
            current_hunk.lines.append((prefix, content))
            
        elif line.strip() == '':
            # Empty lines are allowed between hunks
            continue
            
        else:
            # In V4A format, lines without recognized prefixes might be errors
            # or could be part of file content in some implementations
            if current_hunk is not None:
                # Treat as context line without prefix (some implementations allow this)
                current_hunk.lines.append((' ', line))
    
    # Don't forget the last hunk
    if current_hunk is not None:
        hunks.append(current_hunk)
    
    return hunks


def apply_hunk_create_mode(hunk: DiffHunk, result_lines: List[str]) -> None:
    """
    Apply a hunk in create mode (building a new file).
    
    In create mode:
    - '+' lines are added to the result
    - '-' lines are errors (can't remove from empty file)
    - ' ' context lines are ignored (no context to match)
    
    Args:
        hunk: The hunk to apply
        result_lines: The accumulating result lines (modified in place)
        
    Raises:
        DiffApplicationError: If a removal line is encountered
    """
    for prefix, content in hunk.lines:
        if prefix == '+':
            result_lines.append(content)
        elif prefix == '-':
            raise DiffApplicationError(
                f"Cannot have removal lines in create mode: -{content[:50]}"
            )
        # Context lines (' ') are ignored in create mode


def apply_hunk_update_mode(
    hunk: DiffHunk, 
    content_lines: List[str], 
    current_pos: int
) -> tuple[List[str], int]:
    """
    Apply a hunk in update mode (patching an existing file).
    
    In update mode:
    - ' ' context lines must match the current position
    - '-' lines must match and are removed
    - '+' lines are inserted
    
    Args:
        hunk: The hunk to apply
        content_lines: The current file content lines
        current_pos: Current position in the content
        
    Returns:
        Tuple of (result_lines, new_position)
        
    Raises:
        DiffApplicationError: If context doesn't match or removal fails
    """
    result_lines: List[str] = []
    pos = current_pos
    
    for prefix, content in hunk.lines:
        if prefix == ' ':
            # Context line - must match current position
            if pos >= len(content_lines):
                raise DiffApplicationError(
                    f"Context line extends past end of file: {content[:50]}"
                )
            if content_lines[pos] != content:
                raise DiffApplicationError(
                    f"Context mismatch at line {pos + 1}:\n"
                    f"  Expected: {content[:50]}\n"
                    f"  Got: {content_lines[pos][:50]}"
                )
            result_lines.append(content)
            pos += 1
            
        elif prefix == '-':
            # Removal line - must match current position
            if pos >= len(content_lines):
                raise DiffApplicationError(
                    f"Removal line extends past end of file: -{content[:50]}"
                )
            if content_lines[pos] != content:
                raise DiffApplicationError(
                    f"Removal mismatch at line {pos + 1}:\n"
                    f"  Expected to remove: {content[:50]}\n"
                    f"  Got: {content_lines[pos][:50]}"
                )
            # Don't add to result (it's being removed)
            pos += 1
            
        elif prefix == '+':
            # Addition line - insert at current position
            result_lines.append(content)
            # Don't advance pos (we're inserting, not replacing)
    
    return result_lines, pos


def apply_diff(
    current_content: str,
    diff: str,
    mode: Optional[Literal["create"]] = None
) -> str:
    """
    Apply a V4A diff to content.
    
    This is the main entry point that mirrors the openai-agents SDK's apply_diff function.
    
    Args:
        current_content: The current file content (empty string for create mode)
        diff: The V4A diff text
        mode: "create" for new files, None for updates
        
    Returns:
        The patched content
        
    Raises:
        DiffApplicationError: If the diff cannot be applied
        
    Examples:
        >>> # Create a new file
        >>> diff = '''@@
        ... +def hello():
        ... +    print("Hello, world!")
        ... '''
        >>> content = apply_diff("", diff, mode="create")
        
        >>> # Update existing file
        >>> current = "def hello():\\n    print('Hi')"
        >>> diff = '''@@
        ...  def hello():
        ... -    print('Hi')
        ... +    print("Hello, world!")
        ... '''
        >>> new_content = apply_diff(current, diff)
    """
    if not diff.strip():
        # Empty diff returns original content
        return current_content
    
    # Parse the diff into hunks
    try:
        hunks = parse_v4a_diff(diff)
    except Exception as e:
        raise DiffApplicationError(f"Failed to parse diff: {e}")
    
    if not hunks:
        # No hunks means no changes
        return current_content
    
    # Apply in create mode
    if mode == "create":
        result_lines: List[str] = []
        for hunk in hunks:
            apply_hunk_create_mode(hunk, result_lines)
        return '\n'.join(result_lines)
    
    # Apply in update mode
    content_lines = current_content.split('\n')
    result_lines: List[str] = []
    current_pos = 0
    
    for hunk_idx, hunk in enumerate(hunks):
        # For multi-hunk diffs, we need to search for where this hunk applies
        # We look for the first context line to find the hunk location
        
        # Find the first context or removal line to use as anchor
        anchor_line = None
        for prefix, content in hunk.lines:
            if prefix in (' ', '-'):
                anchor_line = content
                break
        
        if anchor_line is not None:
            # Search for the anchor starting from current position
            found_pos = None
            for search_pos in range(current_pos, len(content_lines)):
                if content_lines[search_pos] == anchor_line:
                    found_pos = search_pos
                    break
            
            if found_pos is None:
                raise DiffApplicationError(
                    f"Failed to apply hunk {hunk_idx + 1}: Could not find context line: {anchor_line[:50]}"
                )
            
            # Add any lines between hunks
            if found_pos > current_pos:
                result_lines.extend(content_lines[current_pos:found_pos])
                current_pos = found_pos
        
        try:
            hunk_result, new_pos = apply_hunk_update_mode(hunk, content_lines, current_pos)
            result_lines.extend(hunk_result)
            current_pos = new_pos
        except DiffApplicationError as e:
            raise DiffApplicationError(f"Failed to apply hunk {hunk_idx + 1}: {e}")
    
    # Append any remaining lines after all hunks
    if current_pos < len(content_lines):
        result_lines.extend(content_lines[current_pos:])
    
    return '\n'.join(result_lines)


def apply_patch_to_file(
    file_path: Path,
    diff_path: Path,
    output_path: Optional[Path] = None,
    mode: Optional[Literal["create"]] = None,
    backup: bool = True
) -> None:
    """
    Apply a V4A diff file to a source file.
    
    Args:
        file_path: Path to the file to patch (can be empty for create mode)
        diff_path: Path to the diff file
        output_path: Where to write the result (defaults to file_path)
        mode: "create" for new files, None for updates
        backup: If True and output_path == file_path, create a .bak backup
        
    Raises:
        DiffApplicationError: If the patch cannot be applied
        FileNotFoundError: If input files don't exist
    """
    # Read the diff
    if not diff_path.exists():
        raise FileNotFoundError(f"Diff file not found: {diff_path}")
    
    diff_text = diff_path.read_text(encoding='utf-8')
    
    # Read current content
    if mode == "create":
        current_content = ""
    else:
        if not file_path.exists():
            raise FileNotFoundError(f"File to patch not found: {file_path}")
        current_content = file_path.read_text(encoding='utf-8')
    
    # Apply the diff
    new_content = apply_diff(current_content, diff_text, mode=mode)
    
    # Determine output location
    out_path = output_path or file_path
    
    # Create backup if needed
    if backup and out_path.exists() and out_path == file_path:
        backup_path = out_path.with_suffix(out_path.suffix + '.bak')
        backup_path.write_text(current_content, encoding='utf-8')
        print(f"Created backup: {backup_path}", file=sys.stderr)
    
    # Write result
    out_path.write_text(new_content, encoding='utf-8')
    print(f"Successfully patched: {out_path}", file=sys.stderr)


def main():
    """Command-line interface for the V4A diff tool."""
    parser = argparse.ArgumentParser(
        description='Apply V4A format diffs to files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create a new file from a diff
  %(prog)s --diff patch.diff --output new_file.py --create
  
  # Update an existing file
  %(prog)s --file source.py --diff changes.diff
  
  # Update with custom output location
  %(prog)s --file source.py --diff changes.diff --output modified.py
  
  # Apply diff from stdin
  cat changes.diff | %(prog)s --file source.py --diff -
        '''
    )
    
    parser.add_argument(
        '--file', '-f',
        type=Path,
        help='File to patch (not needed for --create mode)'
    )
    
    parser.add_argument(
        '--diff', '-d',
        type=Path,
        required=True,
        help='V4A diff file (use "-" for stdin)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file (defaults to input file)'
    )
    
    parser.add_argument(
        '--create', '-c',
        action='store_true',
        help='Create mode: generate a new file from diff'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create .bak backup when overwriting'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.create and not args.file:
        parser.error("--file is required unless using --create mode")
    
    if args.create and not args.output:
        parser.error("--output is required in --create mode")
    
    try:
        # Read diff
        if args.diff == Path('-'):
            diff_text = sys.stdin.read()
        else:
            diff_text = args.diff.read_text(encoding='utf-8')
        
        # Read current content
        if args.create:
            current_content = ""
            file_path = args.output  # Just for display
        else:
            if not args.file.exists():
                print(f"Error: File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            current_content = args.file.read_text(encoding='utf-8')
            file_path = args.file
        
        # Apply diff
        mode = "create" if args.create else None
        new_content = apply_diff(current_content, diff_text, mode=mode)
        
        if args.dry_run:
            print("DRY RUN - No files modified", file=sys.stderr)
            print("\nResult would be:", file=sys.stderr)
            print("-" * 60)
            print(new_content)
            print("-" * 60)
        else:
            # Determine output
            output_path = args.output or args.file
            
            # Backup if needed
            if not args.no_backup and output_path.exists() and output_path == args.file:
                backup_path = output_path.with_suffix(output_path.suffix + '.bak')
                backup_path.write_text(current_content, encoding='utf-8')
                print(f"Created backup: {backup_path}", file=sys.stderr)
            
            # Write result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(new_content, encoding='utf-8')
            
            print(f"Successfully applied diff to: {output_path}", file=sys.stderr)
    
    except DiffApplicationError as e:
        print(f"Error applying diff: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()