import os
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, make_dataclass
from typing import Callable, Literal, Sequence, Optional

# --- CORE DIFF ENGINE ---

ApplyDiffMode = Literal["default", "create"]

@dataclass
class Chunk:
    orig_index: int
    del_lines: list[str]
    ins_lines: list[str]

@dataclass
class ParserState:
    lines: list[str]
    index: int = 0
    fuzz: int = 0

@dataclass
class ReadSectionResult:
    next_context: list[str]
    section_chunks: list[Chunk]
    end_index: int
    eof: bool

# Correctly define the Match object
Match = make_dataclass("Match", [("new_index", int), ("fuzz", int)])

END_PATCH = "*** End Patch"
END_FILE = "*** End of File"
SECTION_TERMINATORS = [END_PATCH, "*** Update File:", "*** Delete File:", "*** Add File:"]
END_SECTION_MARKERS = [*SECTION_TERMINATORS, END_FILE]

def apply_diff(input_str: str, diff: str, mode: ApplyDiffMode = "default") -> str:
    newline = _detect_newline(input_str, diff, mode)
    diff_lines = _normalize_diff_lines(diff)
    if mode == "create":
        return _parse_create_diff(diff_lines, newline=newline)

    normalized_input = input_str.replace("\r\n", "\n")
    parsed_chunks, total_fuzz = _parse_update_diff(diff_lines, normalized_input)
    return _apply_chunks(normalized_input, parsed_chunks, newline=newline)

def _normalize_diff_lines(diff: str) -> list[str]:
    lines = [line.rstrip("\r") for line in re.split(r"\r?\n", diff)]
    if lines and lines[-1] == "":
        lines.pop()
    return lines

def _detect_newline(input_str: str, diff: str, mode: ApplyDiffMode) -> str:
    text = input_str if mode != "create" and "\n" in input_str else diff
    return "\r\n" if "\r\n" in text else "\n"

def _is_done(state: ParserState, prefixes: Sequence[str]) -> bool:
    if state.index >= len(state.lines): return True
    return any(state.lines[state.index].startswith(p) for p in prefixes)

def _read_str(state: ParserState, prefix: str) -> str:
    if state.index >= len(state.lines): return ""
    current = state.lines[state.index]
    if current.startswith(prefix):
        state.index += 1
        return current[len(prefix) :]
    return ""

def _parse_create_diff(lines: list[str], newline: str) -> str:
    parser = ParserState(lines=[*lines, END_PATCH])
    output = []
    while not _is_done(parser, SECTION_TERMINATORS):
        line = parser.lines[parser.index]
        parser.index += 1
        if not line.startswith("+"): raise ValueError(f"Invalid Add Line: {line}")
        output.append(line[1:])
    return newline.join(output)

def _parse_update_diff(lines: list[str], input_str: str):
    parser = ParserState(lines=[*lines, END_PATCH])
    input_lines = input_str.split("\n")
    chunks, cursor = [], 0

    while not _is_done(parser, END_SECTION_MARKERS):
        anchor = _read_str(parser, "@@ ")
        if anchor == "" and parser.index < len(parser.lines) and parser.lines[parser.index] == "@@":
            parser.index += 1
        
        if anchor.strip():
            cursor = _advance_cursor(anchor, input_lines, cursor, parser)

        section = _read_section(parser.lines, parser.index)
        match = _find_context(input_lines, section.next_context, cursor, section.eof)
        
        if match.new_index == -1:
            raise ValueError(f"Context match failed. Could not find context lines in the file.")

        cursor = match.new_index + len(section.next_context)
        parser.fuzz += match.fuzz
        parser.index = section.end_index

        for ch in section.section_chunks:
            chunks.append(Chunk(orig_index=ch.orig_index + match.new_index,
                                del_lines=list(ch.del_lines), ins_lines=list(ch.ins_lines)))
    return chunks, parser.fuzz

def _advance_cursor(anchor: str, lines: list[str], cursor: int, parser: ParserState) -> int:
    for i in range(cursor, len(lines)):
        if lines[i].strip() == anchor.strip():
            return i + 1
    return cursor

def _read_section(lines: list[str], start_index: int) -> ReadSectionResult:
    context, del_lines, ins_lines, chunks = [], [], [], []
    mode, index = "keep", start_index
    
    while index < len(lines):
        raw = lines[index]
        if any(raw.startswith(term) for term in END_SECTION_MARKERS) or raw == "***":
            break
        index += 1
        line = raw if raw else " "
        prefix = line[0]
        
        last_mode, mode = mode, {"+": "add", "-": "delete", " ": "keep"}.get(prefix, "keep")
        if mode == "keep" and last_mode != "keep" and (del_lines or ins_lines):
            chunks.append(Chunk(len(context)-len(del_lines), list(del_lines), list(ins_lines)))
            del_lines, ins_lines = [], []
        
        if mode == "delete": del_lines.append(line[1:]); context.append(line[1:])
        elif mode == "add": ins_lines.append(line[1:])
        else: context.append(line[1:])

    if del_lines or ins_lines:
        chunks.append(Chunk(len(context)-len(del_lines), list(del_lines), list(ins_lines)))
    
    is_eof = index < len(lines) and lines[index] == END_FILE
    return ReadSectionResult(context, chunks, index + (1 if is_eof else 0), is_eof)

def _find_context(lines: list[str], context: list[str], start: int, eof: bool) -> Match:
    # If there's no context, just return current cursor
    if not context:
        return Match(start, 0)
    
    for i in range(start, len(lines) - len(context) + 1):
        if all(lines[i+j].strip() == context[j].strip() for j in range(len(context))):
            return Match(i, 0)
    return Match(-1, 0)

def _apply_chunks(input_str: str, chunks: list[Chunk], newline: str) -> str:
    orig_lines = input_str.split("\n")
    dest_lines, cursor = [], 0
    for chunk in chunks:
        dest_lines.extend(orig_lines[cursor : chunk.orig_index])
        dest_lines.extend(chunk.ins_lines)
        cursor = chunk.orig_index + len(chunk.del_lines)
    dest_lines.extend(orig_lines[cursor:])
    return newline.join(dest_lines)

# --- WORKSPACE & CLI WRAPPER ---

class StandaloneEditor:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if not str(target).startswith(str(self.root)):
            raise RuntimeError(f"Path outside workspace: {path}")
        return target

    def apply(self, op_type: str, path: str, diff_content: str):
        target = self._resolve(path)
        if op_type == "create":
            target.parent.mkdir(parents=True, exist_ok=True)
            content = apply_diff("", diff_content, mode="create")
            target.write_text(content, encoding="utf-8")
            print(f"Successfully Created: {path}")
        elif op_type == "update":
            if not target.exists():
                raise FileNotFoundError(f"File not found: {path}")
            original = target.read_text(encoding="utf-8")
            patched = apply_diff(original, diff_content)
            target.write_text(patched, encoding="utf-8")
            print(f"Successfully Updated: {path}")
        elif op_type == "delete":
            target.unlink(missing_ok=True)
            print(f"Successfully Deleted: {path}")

def main():
    parser = argparse.ArgumentParser(description="Standalone V4A Patch Tool")
    parser.add_argument("type", choices=["create", "update", "delete"], help="Operation type")
    parser.add_argument("path", help="Relative path to the file")
    parser.add_argument("--diff", help="The diff string (or file path containing the diff)")
    parser.add_argument("--root", default=".", help="Workspace root directory")

    args = parser.parse_args()
    editor = StandaloneEditor(Path(args.root))
    
    diff_data = args.diff
    if diff_data:
        if os.path.exists(diff_data):
            diff_data = Path(diff_data).read_text(encoding="utf-8")
        else:
            # FIX: Handle literal \n strings from bash
            diff_data = diff_data.replace("\\n", "\n")

    try:
        editor.apply(args.type, args.path, diff_data or "")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()