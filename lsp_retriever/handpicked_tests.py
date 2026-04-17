import json
import os
from lsp_retriever.search_engine import LSPSearchEngine

def test_handpicked():
    intent_model = "models/intent_classifier_final"
    ner_model = "models/ner_model_final"
    
    engine = LSPSearchEngine(intent_model, ner_model)
    
    # Handpicked complex cases
    cases = [
        {
            "name": "Cross-file Module Reference",
            "query": "Where is the fetch_data module?",
            "expected_file": "data_pipeline/fetch_data.py"
        },
        {
            "name": "Cross-file Usage (Internal)",
            "query": "Where is get_parsers used?",
            "note": "Should find usages in retriever/indexer.py"
        },
        {
            "name": "Class Instantiation across files",
            "query": "Where is SearchEngine used?",
            "expected_files": ["src/main.py", "full_pipeline_demo.py"]
        }
    ]
    
    for case in cases:
        print(f"\nTEST CASE: {case['name']}")
        print(f"Query: {case['query']}")
        res = engine.query(case['query'], verbose=True)
        if not res:
            print("  Result: FAIL (No results found)")
        else:
            print(f"  Result: SUCCESS ({len(res)} results)")
            for r in res[:5]:
                print(f"    - {r['name']} in {r['file_path']}:{r['start_line']}")
                
    engine.close()

if __name__ == "__main__":
    test_handpicked()
