import httpx, os
import ast
from io import BytesIO
from zipfile import ZipFile
from app.indexing.schemas import File, CodeElement
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


MAX_FILE_BYTES = 1_000_000  # 1 MB cap per file (adjust)
DEFAULT_EXTS = {".py",".md"}
BASE_URL = 'https://codeload.github.com'


class GitHubParser:
    """Parser for extracting and processing code from GitHub repositories.
    
    This class downloads GitHub repositories as ZIP files and parses their contents
    into structured CodeElement objects suitable for indexing and retrieval.
    
    Attributes:
        owner (str): GitHub repository owner/organization name
        repo (str): Repository name
        ref (str | None): Git reference (branch, tag, or commit), None for default branch
        
    Example:
        parser = GitHubParser("https://github.com/owner/repo/tree/main")
        code_elements = parser.parse_repo()
        
    Note:
        Supports Python (.py) and Markdown (.md) files only.
        Files larger than MAX_FILE_BYTES are skipped.
    """

    def __init__(self, github_url):
        self.owner, self.repo, self.ref = self.parse_url(github_url)

    def parse_url(self, url: str) -> tuple[str, str, str | None]:
        """Parse a GitHub URL to extract owner, repository name, and optional reference.
        
        Args:
            url: GitHub URL (e.g., 'https://github.com/owner/repo' or 'https://github.com/owner/repo/tree/branch')
            
        Returns:
            Tuple containing (owner, repo, ref) where ref is None if not specified in URL
            
        Raises:
            ValueError: If URL is not a valid GitHub URL or doesn't contain owner/repo
        """
        
        try:
            p = urlparse(url)
        except Exception as e:
            logging.error(f"Error parsing the URL: {str(e)}")
            raise e
        
        if p.netloc.lower() != "github.com":
            raise ValueError("Only github.com URLs are supported")

        parts = [x for x in p.path.strip("/").split("/") if x]
        if len(parts) < 2:
            raise ValueError("URL must be of the form github.com/<owner>/<repo>[/...]")

        owner, repo = parts[0], parts[1].removesuffix(".git")
        ref = None
        if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
            ref = parts[3]
        return owner, repo, ref

    def fetch_repo_zip(self, timeout: float = 60.0) -> bytes:
        """Download this GitHub repository as a ZIP file.
        
        Args:
            timeout: Request timeout in seconds (default: 60.0)
            
        Returns:
            Raw ZIP file content as bytes
            
        Raises:
            ConnectionError: If repository cannot be downloaded (not found, private, or network error)
            
        Note:
            Uses the repository's owner, repo, and ref attributes set during initialization.
            If no ref is specified, tries 'main' then 'master' branches.
        """
        refs_to_try = [self.ref] if self.ref else ["main", "master"]
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            for r in refs_to_try:
                url = f"{BASE_URL}/{self.owner}/{self.repo}/zip/{r}"
                resp = client.get(url)
                if resp.status_code == 200:
                    return resp.content
        raise ConnectionError("Could not download ZIP (ref not found or repo private).")
    
    def get_files_from_zip(self, zip_bytes: bytes, max_bytes: int = MAX_FILE_BYTES) -> list[File]:
        """Extract and process files from a ZIP archive.
        
        Args:
            zip_bytes: Raw ZIP file content as bytes
            max_bytes: Maximum file size in bytes to process (default: MAX_FILE_BYTES)
            
        Returns:
            List of File objects containing content, path, and extension for each processed file
            
        Note:
            Only processes files with extensions in DEFAULT_EXTS (.py, .md).
            Skips directories and files exceeding max_bytes limit.
            Text encoding falls back from UTF-8 to Latin-1 if decoding fails.
        """
        files = []
        with ZipFile(BytesIO(zip_bytes)) as zip_file:
      
            prefix = os.path.commonpath([i.filename for i in zip_file.infolist()]) + "/"
            for info in zip_file.infolist():
                if info.is_dir() or info.file_size > max_bytes:
                    continue
                inner = info.filename
                if not inner.startswith(prefix):
                    continue
                rel = inner[len(prefix):]  # repo-relative
                ext = os.path.splitext(rel)[1].lower()
                if ext not in DEFAULT_EXTS:
                    continue
                # Read & decode (assume utf-8; fall back to latin-1 with replacement)
                with zip_file.open(info) as f:
                    raw = f.read()
                try:
                    text = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    text = raw.decode("latin-1", errors="replace").strip()

                file = File(content=text, path=rel, extension=ext)
                files.append(file)
        return files
    
    def parse_code_simple(self, file: File, max_lines_per_elem: int = 100) -> list[CodeElement]:
  
        try:
            tree = ast.parse(file.content)
        except Exception:
            return []

        source = file.path
        extension = file.extension
        lines = file.content.splitlines()
        lines = [line[: 200] + '\n' for line in lines]

        def slice_node(node: ast.AST) -> list[str]:
            """Extract source lines for an AST node, including decorators."""
            # Find the earliest line (decorators come before the node itself)
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            end = getattr(node, "end_lineno", node.lineno)  # fallback if end_lineno missing
            return lines[start-1:end]  # Convert to 0-based indexing
        
        headers: list[str] = []
        code_elements: list[CodeElement] = []
        previous_text: list[str] = []

        for node in tree.body:  # top-level order
            node_text = slice_node(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if previous_text and (len(previous_text) + len(node_text)) > max_lines_per_elem:
                    code_elements.append(
                        CodeElement(
                            text=''.join(previous_text[:200]).strip(), 
                            source=source, 
                            header=''.join(headers).strip() if headers else None,
                            extension=extension
                        )
                    )
                    previous_text = []

                previous_text = previous_text + ["\n"] + node_text if previous_text else node_text
            else:
                headers.extend(node_text)

        # Emit leftover (may be < min_lines)
        if previous_text:
            code_elements.append(
                CodeElement(
                    text=''.join(previous_text).strip(), 
                    source=source, 
                    header=''.join(headers).strip() if headers else None,
                    extension=extension
                )
            )

        # If no defs/classes at all, return whole file as one element with whatever header we collected
        if not code_elements:
            code_elements.append(CodeElement(text=file.content.strip(), source=source, extension=extension))

        return code_elements
    
    def parse_code(self, file: File, max_lines_per_elem: int = 150) -> list[CodeElement]:
        """Parse Python code into structured CodeElement objects with intelligent chunking.
        
        Args:
            file: File object containing Python source code to parse
            max_lines_per_elem: Maximum lines per code element before splitting (default: 100)
            
        Returns:
            List of CodeElement objects, each containing logically grouped code chunks
            
        Note:
            Intelligently splits large classes into multiple elements while preserving context.
            Groups related functions and maintains header context with imports/globals.
            Uses AST parsing to respect Python structure rather than arbitrary line splits.
        """
  
        try:
            tree = ast.parse(file.content)
        except Exception:
            return []

        source = file.path
        extension = file.extension
        lines = file.content.splitlines()
        lines = [line[: 200] + '\n' for line in lines]

        def slice_node(node: ast.AST) -> list[str]:
            """Extract source lines for an AST node, including decorators."""
            # Find the earliest line (decorators come before the node itself)
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            end = getattr(node, "end_lineno", node.lineno)  # fallback if end_lineno missing
            return lines[start-1:end]  # Convert to 0-based indexing
        
        def split_class(node: ast.ClassDef) -> list[list[str]]:
            """Split large classes into multiple chunks while preserving structure."""
            class_lines = slice_node(node)
            # If class fits within limit, return as single chunk
            if len(class_lines) <= max_lines_per_elem:
                return [class_lines]
            
            # Split large class into multiple parts
            class_parts = []
            part = [f'class {node.name}:\n']  # Start each part with class header
            
            for sub_node in node.body:
                sub_node_lines = slice_node(sub_node)
                
                # If adding this method/attribute would exceed limit, finalize current part
                if len(part) + len(sub_node_lines) > max_lines_per_elem and len(part) > 1:
                    part.append('    ...\n')  # Indicate continuation
                    class_parts.append(part)
                    # Start new part with class header and continuation marker
                    part = [f'class {node.name}:\n    ...\n']

                part.extend(sub_node_lines)

            # Add final part if it has content beyond just the header
            if len(part) > 1:
                class_parts.append(part)
            return class_parts

        headers: list[str] = []
        code_elements: list[CodeElement] = []
        previous_text: list[str] = []

        for node in tree.body:  # top-level order
            node_text = slice_node(node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if previous_text and (len(previous_text) + len(node_text)) > max_lines_per_elem:
                    code_elements.append(
                        CodeElement(
                            text=''.join(previous_text).strip(), 
                            source=source, 
                            header=''.join(headers).strip() if headers else None,
                            extension=extension
                        )
                    )
                    previous_text = []

                previous_text = previous_text + ["\n"] + node_text if previous_text else node_text
            elif isinstance(node, ast.ClassDef):
                class_parts = split_class(node)
                for part in class_parts:
                    if previous_text and (len(previous_text) + len(part)) > max_lines_per_elem:
                        code_elements.append(
                            CodeElement(
                                text=''.join(previous_text).strip(), 
                                source=source, 
                                header=''.join(headers).strip() if headers else None,
                                extension=extension
                            )
                        )
                        previous_text = []
                    previous_text = previous_text + ["\n"] + part if previous_text else part
            else:
                headers.extend(node_text)

        # Emit leftover (may be < min_lines)
        if previous_text:
            code_elements.append(
                CodeElement(
                    text=''.join(previous_text).strip(), 
                    source=source, 
                    header=''.join(headers).strip() if headers else None,
                    extension=extension
                )
            )

        # If no defs/classes at all, return whole file as one element with whatever header we collected
        if not code_elements:
            code_elements.append(CodeElement(text=file.content.strip(), source=source, extension=extension))

        return code_elements
    
    def parse_markdown(self, file: File, min_lines_per_elem: int = 100, overlap_lines: int = 5) -> list[CodeElement]:
        """Parse Markdown content into overlapping CodeElement chunks.
        
        Args:
            file: File object containing Markdown content to parse
            min_lines_per_elem: Lines per chunk (default: 100)
            overlap_lines: Number of overlapping lines between chunks (default: 5)
            
        Returns:
            List of CodeElement objects with chunked Markdown content
            
        Note:
            Creates overlapping chunks to preserve context across boundaries.
            Step size = min_lines_per_elem - overlap_lines to ensure forward progress.
            Overlap is clamped to be less than chunk size to avoid infinite loops.
        """
        source = file.path
        lines = file.content.splitlines(keepends=True)
        extension = file.extension
        num_lines = len(lines)

        # Clamp overlap and compute step (so we move forward but keep overlap)
        overlap_lines = max(0, min(overlap_lines, min_lines_per_elem - 1))
        step = max(1, min_lines_per_elem - overlap_lines)

        chunks: list[CodeElement] = []
        for start in range(0, num_lines, step):
            end = start + min_lines_per_elem
            chunk_text = "".join(lines[start:end])
            chunks.append(CodeElement(text=chunk_text, source=source, extension=extension))

        return chunks
    
    def parse_repo(self) -> list[CodeElement]:
        """Parse the GitHub repository into structured code elements.
        
        Returns:
            List of CodeElement objects containing parsed content from Python and Markdown files
            
        Note:
            Downloads the repository ZIP using instance attributes (owner, repo, ref).
            Processes .py files using AST parsing and .md files using chunk-based parsing.
            Filters files based on DEFAULT_EXTS and MAX_FILE_BYTES limits.
        """
        zip_bytes = self.fetch_repo_zip()
        files = self.get_files_from_zip(zip_bytes)

        code_elements = []
        for file in files:
            if file.extension == '.py':
                code_elements.extend(self.parse_code(file))
            if file.extension == '.md':
                code_elements.extend(self.parse_markdown(file))
            
        return code_elements
