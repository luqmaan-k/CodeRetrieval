from tree_sitter import Language, Parser
from pathlib import Path

# Path to the compiled library
LIB_PATH = Path("retriever/build/my-languages.so")
GRAMMAR_DIR = Path("retriever/grammars")

def setup_parsers():
    # Ensure build dir exists
    LIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Python and Java
    # Note: Modern tree-sitter python package (tree-sitter-python) 
    # usually provides the language object directly.
    # However, for a custom build:
    try:
        import tree_sitter_python as tspython
        import tree_sitter_java as tsjava
        
        # In newer tree-sitter, you can often just do:
        # PY_LANGUAGE = Language(tspython.language())
        print("Using pre-installed tree-sitter language packages.")
    except ImportError:
        print("Falling back to manual build (if needed)...")

if __name__ == "__main__":
    setup_parsers()
