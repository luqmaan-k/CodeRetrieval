from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
import tree_sitter_cpp as tscpp
import tree_sitter_c_sharp as tscsharp

def get_languages():
    """
    Returns a dictionary of Language objects for all supported languages.
    """
    return {
        'python': Language(tspython.language()),
        'java': Language(tsjava.language()),
        'javascript': Language(tsjs.language()),
        'cpp': Language(tscpp.language()),
        'c_sharp': Language(tscsharp.language())
    }

def get_parsers():
    """
    Initializes and returns Tree-sitter parsers for all supported languages.
    """
    languages = get_languages()
    return {lang: Parser(obj) for lang, obj in languages.items()}

if __name__ == "__main__":
    parsers = get_parsers()
    for lang in parsers:
        print(f"Parser initialized successfully for {lang}.")
