import os
import json
from pathlib import Path
from tree_sitter import Query, QueryCursor
from retriever.parser_setup import get_parsers, get_languages

# Mapping of file extensions to supported languages
EXTENSION_TO_LANGUAGE = {
    '.py': 'python',
    '.java': 'java',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.h': 'cpp',
    '.hpp': 'cpp',
    '.cs': 'c_sharp'
}

def load_queries(languages):
    """Loads tag queries for all supported languages."""
    queries = {}
    for lang in languages:
        # Map language name to folder name (if different)
        folder_name = lang.replace('_', '-')
        query_path = Path(f"retriever/grammars/tree-sitter-{folder_name}/queries/tags.scm")
        
        if query_path.exists():
            with open(query_path, 'r') as f:
                queries[lang] = f.read()
        else:
            print(f"Warning: No tags.scm found for {lang} at {query_path}")
            
    return queries

def index_codebase(root_dir):
    """
    Traverses the codebase, parses supported source files, and extracts definitions.
    """
    parsers = get_parsers()
    languages = get_languages()
    query_scms = load_queries(languages.keys())
    
    # Pre-compile queries
    queries = {}
    for lang, scm in query_scms.items():
        try:
            queries[lang] = Query(languages[lang], scm)
        except Exception as e:
            print(f"Error compiling query for {lang}: {e}")
    
    index = []
    
    # Directories to ignore
    IGNORE_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'build', 'dist'}
    
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            
            if ext not in EXTENSION_TO_LANGUAGE:
                continue
                
            lang = EXTENSION_TO_LANGUAGE[ext]
            if lang not in parsers or lang not in queries:
                continue
            
            parser = parsers[lang]
            query = queries[lang]
            
            print(f"Indexing {file_path} ({lang})...")
            
            try:
                with open(file_path, 'rb') as f:
                    code = f.read()
                
                tree = parser.parse(code)
                cursor = QueryCursor(query)
                captures_dict = cursor.captures(tree.root_node)
                
                # Group nodes by their byte range to associate tags
                nodes_by_range = {}
                
                for tag_name, nodes in captures_dict.items():
                    for node in nodes:
                        node_range = (node.start_byte, node.end_byte)
                        if node_range not in nodes_by_range:
                            nodes_by_range[node_range] = {'node': node, 'tags': set()}
                        nodes_by_range[node_range]['tags'].add(tag_name)
                
                for node_range, info in nodes_by_range.items():
                    node = info['node']
                    tags = info['tags']
                    
                    # Look for definition tags
                    definition_tag = next((t for t in tags if t.startswith('definition.')), None)
                    
                    if definition_tag:
                        name = "unknown"
                        if 'name' in tags:
                            name = code[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                        else:
                            # Fallback: look for identifier children
                            for child in node.children:
                                if 'identifier' in child.type or 'name' in child.type:
                                    name = code[child.start_byte:child.end_byte].decode('utf8', errors='ignore')
                                    break
                        
                        index.append({
                            'name': name,
                            'type': definition_tag.split('.')[-1],
                            'language': lang,
                            'file_path': str(file_path),
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1,
                            'start_column': node.start_point[1],
                            'end_column': node.end_point[1]
                        })
                        
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
                
    return index

if __name__ == "__main__":
    PROJECT_ROOT = "."
    OUTPUT_FILE = "data/processed/ast_index.json"
    
    os.makedirs("data/processed", exist_ok=True)
    
    print("Starting AST indexing...")
    full_index = index_codebase(PROJECT_ROOT)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(full_index, f, indent=2)
        
    print(f"Indexing complete. {len(full_index)} definitions stored in {OUTPUT_FILE}")
