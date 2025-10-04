from urllib.parse import urlparse
import os
import httpx  # TODO: Will be used in fetch_repo_zip method
import ast
from io import BytesIO
from zipfile import ZipFile

from app.indexing.schemas import File, CodeElement


BASE_URL = 'https://codeload.github.com'
MAX_FILE_BYTES = 1_000_000  # 1 MB cap per file (adjust)
DEFAULT_EXTS = {".py", ".md"}


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
        owner, repo, ref = '', '', None
        # Implement the parse_url function:
        # - Use the urlparse function to parse the URL.
        # - Make sure the netloc attribute is equal to "github.com". Raise an error if it is not the case
        # - Split the path by the slash character "/".
        # - If the path doesn't contain at least 2 pieces, there is a problem, and we should raise an error in that case.
        # - The first part of the path is the owner, and the second part is the repo name. In the specific case the URL is passed with the .git suffix, we need to remove it. For example, from https://github.com/huggingface/transformers.git , we should extract huggingface and transformers.
        # - If the path contains at least 4 parts and the third part is equal to tree or blob, then the fourth part is the reference. In any other cases, let's set the reference value to None.
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc != "github.com":
                raise ValueError("Invalid GitHub URL")
            path = parsed_url.path.strip("/").split("/")
            if len(path) < 2:
                raise ValueError("Invalid GitHub URL")
            owner, repo = path[0], path[1]
            if len(path) >= 4 and path[2] in ["tree", "blob"]:
                ref = path[3]
        except Exception as e:
            raise ValueError(f"Invalid GitHub URL: {e}")
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
        # Implement fetch_repo_zip function:
        # - If the reference is None, try 'master' and 'main', otherwise use the one existing one.
        # - Construct the URL from the BASE_URL (https://codeload.github.com), the owner, the repo name, and the different references. 
        # - Instantiate the httpx.Client client with follow_redirects=True and timeout, and call client.get on the URL.
        # - If the response from the HTTP call is not 200 (success), then raise a connection error.
        refs_to_try = ['master', 'main'] if self.ref is None else [self.ref]

        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            for ref in refs_to_try:
                url = f"{BASE_URL}/{self.owner}/{self.repo}/zip/{ref}"
                print(f"URL: {url}")
                try:
                  response = client.get(url)
                except Exception as e:
                  print(e)
                print(f"Response status code: {response.status_code}")
                if response.status_code == 200:
                    return response.content
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
            
            # use the os.path.commonpath function to extract the common root path for every file in the zip file. 
            # You can iterate through all the filenames by using:
            # [i.filename for i in zip_file.infolist()]
            # The prefix variable will be the result of that common root path + "/". 
            prefix = os.path.commonpath([i.filename for i in zip_file.infolist()]) + "/"

            for info in zip_file.infolist():
                text = None
                path = None
                extension = None
                # Filter the files by ignoring:
                # - The directories: info.is_dir()
                # - The file names that do not start with the prefix.
                # - The files with more data than MAX_FILE_BYTES: info.file_size > max_bytes.
                # - The non-documentation or Python files: DEFAULT_EXTS = {".py", ".md"}.
                # 
                # You can extract the extensions by using the os.path.splitext function:
                # os.path.splitext(path)[1].lower()
                #
                # For the non-ignored file extract the content:
                # with zip_file.open(info) as f:
                #     raw = f.read()
                #     try:
                #         text = raw.decode("utf-8").strip()
                #     except UnicodeDecodeError:
                #         text = raw.decode("latin-1", errors="replace").strip()
                if not info.is_dir() and info.filename.startswith(prefix) and info.file_size <= max_bytes and os.path.splitext(info.filename)[1].lower() in DEFAULT_EXTS:
                    with zip_file.open(info) as f:
                        raw = f.read()
                        try:
                            text = raw.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            text = raw.decode("latin-1", errors="replace").strip()
                        
                        name = info.filename[len(prefix):]
                        extension = os.path.splitext(name)[1].lower()
                        files.append(File(content=text, path=name, extension=extension))


        # Return a list of File data structures.
        return files
    
    def parse_code(self, file: File, max_lines_per_elem: int = 200) -> list[CodeElement]:
        """Parse Python code into structured CodeElement objects with intelligent chunking.
        
        Args:
            file: File object containing Python source code to parse
            max_lines_per_elem: Maximum lines per code element before splitting (default: 200)
            
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

        # Initializing the variables
        headers: list[str] = []
        code_elements: list[CodeElement] = []
        previous_text: list[str] = []

        for node in tree.body:  # top-level order
            # Use slice_node to extract the lines of the current node
            node_text = slice_node(node)
            # Classify the current top-level node:
            #  - Function / AsyncFunction  -> treat as an atomic code chunk.
            #  - Class                     -> split into bounded parts via split_class(node).
            #  - Anything else             -> treat as header (top-level non-def/class code).
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # If it's a Function/AsyncFunction:
                # - If adding this function to `previous_text` would exceed `max_lines_per_elem`,
                # emit `previous_text` as a CodeElement and clear it (do NOT split the function).
                # - Append the function's lines (`node_text`) to `previous_text` (prefix with "\n" if needed).
                if previous_text and len(previous_text) + len(node_text) > max_lines_per_elem:
                    code_elements.append(
                        CodeElement(
                            header=''.join(headers).strip() if headers else None,
                            source=source,
                            extension=file.extension,
                            description='',
                            text=''.join(previous_text).strip()
                        )
                    )
                    previous_text = []
                previous_text = previous_text + ["\n"] + node_text if previous_text else node_text
               
            else:
                # If it's neither function nor class:
                # - Extend `headers` with `node_text` (we include all top-level non-def/class lines in the header).
                headers.extend(node_text)
                
               
        # Emit leftover (may be < min_lines)
        # If there are some leftovers in previous_text, add a new CodeElement to code_elements.
        if len(previous_text) > 0:
            code_elements.append(CodeElement(
                header=''.join(headers).strip() if headers else None,
                source=source,
                extension=file.extension,
                description='',
                text=''.join(previous_text).strip()
            )
        )

        if not code_elements:
            code_elements.append(CodeElement(
                source=source,
                extension=file.extension,
                description='',
                text=file.content.strip()
            ))

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
        extension = file.extension
        lines = file.content.splitlines(keepends=True)
        num_lines = len(lines)

        # TODO: Clamp overlap_lines to ensure that it is between 0 and min_lines_per_elem - 1.
        overlap_lines = None
        # TODO: Implement the step variable as min_lines_per_elem - overlap_lines.
        step = None

        chunks: list[CodeElement] = []
        for start in range(0, num_lines, step):
            # TODO: Iterate through the lines, and capture a CodeElement without a header for 
            # each chunk of text of size min_lines_per_elem, every step lines.
            pass

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
        # TODO: The main function of the class
        # - Fetch the repo byte data
        # - Extract the files from the zip file
        # - If the extension is .py, then we should extract the code, and if it .md, we should extract the markdown
        # - Return all the code elements from the repo
        code_elements = []         
        return code_elements
