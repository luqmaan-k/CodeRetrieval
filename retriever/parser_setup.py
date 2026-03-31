from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_java as tsjava

def get_parsers():
    """
    Initializes and returns Tree-sitter parsers for Python and Java.
    """
    PY_LANGUAGE = Language(tspython.language())
    JAVA_LANGUAGE = Language(tsjava.language())
    
    py_parser = Parser(PY_LANGUAGE)
    java_parser = Parser(JAVA_LANGUAGE)
    
    return {
        'python': py_parser,
        'java': java_parser
    }

if __name__ == "__main__":
    parsers = get_parsers()
    print("Parsers initialized successfully for Python and Java.")
