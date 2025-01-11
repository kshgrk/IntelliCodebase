import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import vertexai
import datetime

from vertexai.generative_models import (
    Part,
)
from vertexai.preview import caching
from vertexai.preview.generative_models import GenerativeModel

@dataclass
class CodeChunk:
    file_path: str
    start_line: int
    end_line: int
    content: str

@dataclass
class Issue:
    chunk: CodeChunk
    description: str
    fix_suggestion: Optional[str] = None
    priority: int = 0

class CodebaseAnalyzer:
    def __init__(self, base_path: str, chunk_size: int = 1000, ttl: int = 60, min_token_count: int = 32768):
        self.base_path = base_path
        self.chunk_size = chunk_size
        self.issues_db = {}
        self.progress_file = os.path.join(base_path, "analysis_progress.json")
        self.cache_mapping_file = os.path.join(base_path, "cache_mapping.json")
        self.cache_mapping = self.load_cache_mapping()
        self.ttl = ttl
        self.PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.LOCATION = os.getenv("REGION")
        self.min_token_count = min_token_count
        vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)

    def load_progress(self) -> Dict:
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'processed_files': [], 'current_file': None, 'current_position': 0}

    def save_progress(self, progress: Dict):
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

    def load_cache_mapping(self) -> Dict:
        if os.path.exists(self.cache_mapping_file):
            with open(self.cache_mapping_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache_mapping(self):
        with open(self.cache_mapping_file, 'w') as f:
            json.dump(self.cache_mapping, f)

    def get_file_chunks(self, file_path: str) -> List[CodeChunk]:
        chunks = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            print(f"Skipping file {file_path} due to UnicodeDecodeError")
            return []

        current_chunk = []
        current_start = 0

        for i, line in enumerate(lines):
            current_chunk.append(line)
            if len(''.join(current_chunk)) >= self.chunk_size:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    start_line=current_start,
                    end_line=i,
                    content=''.join(current_chunk)
                ))
                current_chunk = []
                current_start = i + 1

        if current_chunk:
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=current_start,
                end_line=len(lines),
                content=''.join(current_chunk)
            ))
        print(f'Found {len(chunks)} chunks in file {file_path}')
        return chunks
    
    def is_cache_valid(self, file_path: str) -> bool:
        """Checks if the cache for the given file is still valid based on modification time."""
        if "codebase_cache_name" not in self.cache_mapping:
            return False

        try:
            last_modified_time = os.path.getmtime(file_path)
        except OSError:
            # Handle cases where the file might not exist
            return False

        cache_creation_time = datetime.datetime.fromisoformat(self.cache_mapping["codebase_cache_name"].split(".")[-1])
        
        # Convert to Unix timestamp for comparison
        cache_creation_timestamp = cache_creation_time.timestamp()
        
        return last_modified_time <= cache_creation_timestamp

    def analyze_chunk(self, chunk: CodeChunk, model) -> List[Issue]:
        # Retrieve CachedContent
        cached_content_name = self.cache_mapping.get("codebase_cache_name")

        if cached_content_name:
            # Use cached model
            cached_content = caching.CachedContent(cached_content_name=cached_content_name)
            cached_model = GenerativeModel.from_cached_content(
                cached_content=cached_content
            )
        else:
            # Fallback to the original model if no cache exists
            cached_model = model
            print(
                "Warning: No cache found. Using the original model for analysis."
            )

        chunk_content = chunk.content
        if "--- PADDING START ---" in chunk.content and "--- PADDING END ---" in chunk.content:
            chunk_content = chunk_content.split("--- PADDING START ---")[0] + chunk_content.split("--- PADDING END ---")[1]

        # Construct prompt for the LLM
        prompt = f"""Analyze the following code chunk for issues:
        File: {chunk.file_path}
        Lines: {chunk.start_line}-{chunk.end_line}

        ```
        {chunk_content}

        ```

        Identify any issues and suggest fixes. Return the response in the format:
        Issue: <Description of the issue>
        Fix Suggestion: <Suggested fix, or None if no fix is suggested>
        Priority: <Priority of the issue, integer>
        ---
        If no issue found then return:
        No issues found
        """

        # Get LLM response using the appropriate model
        try:
            response = cached_model.generate_content(prompt)
            print(f"Response text: {response.text}")
        except Exception as e:
            print(f"Error during content generation: {e}")
            return []

        issues = []
        # Parse response only if the response object has candidates
        if response.candidates:
            # Assuming 'text' is an attribute of the response object
            text = response.text
            if "No issues found" not in text:
                # Parse response and extract issues
                for issue_text in text.split("---"):
                    issue_text = issue_text.strip()
                    if issue_text:
                        try:
                            issue_parts = issue_text.split("\n")
                            description = issue_parts[0].split(":", 1)[1].strip()
                            fix_suggestion = None
                            priority = 0
                            for part in issue_parts[1:]:
                                if part.startswith("Fix Suggestion:"):
                                    fix_suggestion = part.split(":", 1)[1].strip()
                                    if fix_suggestion == "None":
                                        fix_suggestion = None
                                elif part.startswith("Priority:"):
                                    priority = int(part.split(":", 1)[1].strip())
                            issues.append(
                                Issue(chunk, description, fix_suggestion, priority)
                            )
                        except Exception as e:
                            print(f"Error parsing issue: {e}")

        return issues

    def process_codebase(self, model):
        progress = self.load_progress()
        self.issues_db = {}  # Reset issues_db

        # Aggregate content for all files recursively
        codebase_content_parts = []
        codebase_files = []  # Keep track of files added to the cache

        def process_directory(directory):
            nonlocal codebase_content_parts, codebase_files
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith((".py", ".js", ".cpp")):
                        file_path = os.path.join(root, file)
                        if file_path not in progress["processed_files"]:
                            try:
                                with open(file_path, "r") as f:
                                    file_content = f.read()
                                codebase_content_parts.append(Part.from_text(file_content))
                                codebase_files.append(file_path)  # Add file to the list
                            except UnicodeDecodeError:
                                print(f"Skipping file {file_path} due to UnicodeDecodeError")

        process_directory(self.base_path)  # Start recursive processing from the base path

        # --- Padding Logic ---
        if codebase_content_parts:
            total_tokens = sum(len(part.text.split()) for part in codebase_content_parts)  # Estimate token count

            if total_tokens < self.min_token_count:
                padding_tokens_needed = self.min_token_count - total_tokens
                # Create padding content that is unlikely to interfere with analysis
                padding_content = "This is padding content to meet the minimum token requirement for caching. " * (padding_tokens_needed // 10 + 1) 
                # Add a unique identifier to easily recognize and ignore padding later
                padding_content = f"--- PADDING START ---\n{padding_content}\n--- PADDING END ---"
                codebase_content_parts.append(Part.from_text(padding_content))
                print(f"Added padding of {len(padding_content.split())} tokens.")

        # Create CachedContent for the entire codebase if not exists or if cache is invalid
        if codebase_content_parts and ("codebase_cache_name" not in self.cache_mapping or not self.is_cache_valid(codebase_files[0])):
            cached_content = caching.CachedContent.create(
                model_name="gemini-1.5-pro-002",
                system_instruction="Analyze the following code for issues and suggest fixes.",
                contents=codebase_content_parts,
                ttl=datetime.timedelta(minutes=self.ttl),
                display_name=f"cache_for_codebase_{self.base_path.replace('/', '_')}",
            )
            self.cache_mapping["codebase_cache_name"] = cached_content.name
            self.save_cache_mapping()

        # Analyze each file using the cached content
        for file_path in codebase_files:
            chunks = self.get_file_chunks(file_path)
            for chunk in chunks:
                # In analyze_chunk, make sure to ignore padding content when presenting to the model
                issues = self.analyze_chunk(chunk, model)
                if issues:
                    for issue in issues:
                        self.issues_db.setdefault(file_path, []).append(
                            {
                                "issue": issue,
                                "cached_content_name": self.cache_mapping.get("codebase_cache_name"),
                            }
                        )
            print(issues)

            progress["processed_files"].append(file_path)
            self.save_progress(progress)
