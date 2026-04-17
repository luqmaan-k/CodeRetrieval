import sys
import os
from pathlib import Path
from lsp_retriever.search_engine import LSPSearchEngine

def print_result(result):
    print(f"\n--- Result: {result['name']} ({result['type']}) ---")
    print(f"File: {result['file_path']}, Line: {result['start_line']}")
    print(f"Match Score: {result.get('match_score', 'N/A')}")
    
    try:
        # Resolve full path
        full_path = Path(result['file_path'])
        if not full_path.is_absolute():
            full_path = Path(".").absolute() / full_path
            
        with open(full_path, 'r') as f:
            lines = f.readlines()
            start = result['start_line'] - 1
            end = min(result['end_line'], start + 10) # Print up to 10 lines
            snippet = "".join(lines[start:end])
            print("\nCode Snippet:")
            print("-" * 20)
            print(snippet)
            if result['end_line'] > start + 10:
                print("... (truncated)")
            print("-" * 20)
    except Exception as e:
        print(f"Could not read snippet from {result['file_path']}: {e}")

def main():
    intent_model = "models/intent_classifier_final"
    ner_model = "models/ner_model_final"
    
    if not Path(intent_model).exists() or not Path(ner_model).exists():
        print("Error: NLP models not found. Please run the training scripts first.")
        sys.exit(1)
        
    engine = LSPSearchEngine(intent_model, ner_model)
    
    print("\n" + "=" * 40)
    print("LSP-based Semantic Code Retrieval CLI")
    print("Type 'exit' to quit.")
    print("=" * 40)
    
    while True:
        try:
            query_str = input("\nEnter search query: ").strip()
            if query_str.lower() in ['exit', 'quit']:
                break
            if not query_str:
                continue
                
            results = engine.query(query_str, verbose=True)
            
            if not results:
                print("No results found.")
            else:
                for res in results:
                    print_result(res)
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            
    engine.close()

if __name__ == "__main__":
    main()
