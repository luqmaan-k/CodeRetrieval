import os
import json
from pathlib import Path
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python as tspython
import tree_sitter_java as tsjava

def load_queries():
    """Loads tag queries for Python and Java."""
    py_query_path = Path("retriever/grammars/tree-sitter-python/queries/tags.scm")
    java_query_path = Path("retriever/grammars/tree-sitter-java/queries/tags.scm")
    
    with open(py_query_path, 'r') as f:
        py_query_scm = f.read()
    with open(java_query_path, 'r') as f:
        java_query_scm = f.read()
        
    return py_query_scm, java_query_scm

def index_codebase(root_dir):
    """
    Traverses the codebase, parses Python and Java files, and extracts definitions.
    """
    PY_LANGUAGE = Language(tspython.language())
    JAVA_LANGUAGE = Language(tsjava.language())
    
    py_parser = Parser(PY_LANGUAGE)
    java_parser = Parser(JAVA_LANGUAGE)
    
    py_query_scm, java_query_scm = load_queries()
    py_query = Query(PY_LANGUAGE, py_query_scm)
    java_query = Query(JAVA_LANGUAGE, java_query_scm)
    
    index = []
    
    for root, _, files in os.walk(root_dir):
        # Skip hidden directories and .venv
        if any(part.startswith('.') for part in Path(root).parts) or 'venv' in root:
            continue
            
        for file in files:
            file_path = Path(root) / file
            if file.endswith('.py'):
                lang = 'python'
                parser = py_parser
                query = py_query
            elif file.endswith('.java'):
                lang = 'java'
                parser = java_parser
                query = java_query
            else:
                continue
            
            print(f"Indexing {file_path}...")
            
            try:
                with open(file_path, 'rb') as f:
                    code = f.read()
                
                tree = parser.parse(code)
                # In 0.25.2, QueryCursor needs the query in __init__
                cursor = QueryCursor(query)
                captures_dict = cursor.captures(tree.root_node)
                
                # We need to find definitions and their associated names.
                # In tags.scm, a definition is often tagged like @definition.function
                # and the name is tagged as @name.
                
                # Map all captured nodes by their byte range to associate @name with @definition.*
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
                    
                    definition_tag = next((t for t in tags if t.startswith('definition.')), None)
                    
                    if definition_tag:
                        # If this node itself is tagged @name, great.
                        # Otherwise, we might need to look for a @name tag in the SAME range 
                        # (which we just did by grouping by range).
                        
                        name = "unknown"
                        if 'name' in tags:
                            name = code[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                        else:
                            # If no @name tag on this node, it might be that the @name tag 
                            # was on a child node (e.g. the identifier). 
                            # However, our current grouping only looks at the exact same range.
                            
                            # Let's try to find an identifier child if it's a known definition type
                            for child in node.children:
                                if child.type == 'identifier' or child.type == 'type_identifier':
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
