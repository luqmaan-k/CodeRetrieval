import sys
import os
from pathlib import Path
from retriever.search_engine import SearchEngine

def print_result(result):
    print(f"\n--- Result: {result['name']} ({result['type']}) ---")
    print(f"File: {result['file_path']}, Line: {result['start_line']}")
    print(f"Language: {result['language']}, Match Score: {result.get('match_score', 'N/A')}")
    
    try:
        with open(result['file_path'], 'r') as f:
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
        print(f"Could not read snippet: {e}")

def main():
    # Ensure indices exist
    ast_index = "data/processed/ast_index.json"
    intent_model = "models/intent_classifier_final"
    ner_model = "models/ner_model_final"
    
    if not Path(ast_index).exists():
        print(f"Error: AST index not found at {ast_index}. Please run the indexer first.")
        sys.exit(1)
        
    if not Path(intent_model).exists() or not Path(ner_model).exists():
        print("Error: NLP models not found. Please run the training scripts first.")
        sys.exit(1)
        
    engine = SearchEngine(intent_model, ner_model, ast_index)
    
    print("\n" + "=" * 40)
    print("Semantic Code Retrieval CLI")
    print("Type 'exit' to quit.")
    print("=" * 40)
    
    while True:
        try:
            query_str = input("\nEnter search query: ").strip()
            if query_str.lower() in ['exit', 'quit']:
                break
            if not query_str:
                continue
                
            results = engine.query(query_str)
            
            if not results:
                print("No results found.")
            else:
                for res in results:
                    print_result(res)
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
