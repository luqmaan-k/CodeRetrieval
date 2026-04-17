import time
import json
import os
from retriever.search_engine import SearchEngine
from lsp_retriever.search_engine import LSPSearchEngine

def run_benchmark():
    ast_index = "data/processed/ast_index.json"
    intent_model = "models/intent_classifier_final"
    ner_model = "models/ner_model_final"
    
    # 1. Initialize Engines
    print("Initializing AST-based Search Engine...")
    ast_engine = SearchEngine(intent_model, ner_model, ast_index)
    
    print("\nInitializing LSP-based Search Engine...")
    lsp_engine = LSPSearchEngine(intent_model, ner_model)
    
    # 2. Define Benchmark Queries
    benchmark_queries = [
        "Where is the SearchEngine class defined?",
        "Find the query function",
        "Show me the fetch_data function",
        "find the index_codebase function",
        "Where is main defined?"
    ]
    
    # 3. Execute and Compare
    for q in benchmark_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {q}")
        print("-" * 60)
        
        # AST Search
        start = time.time()
        ast_results = ast_engine.query(q)
        ast_time = (time.time() - start) * 1000
        
        # LSP Search
        start = time.time()
        lsp_results = lsp_engine.query(q)
        lsp_time = (time.time() - start) * 1000
        
        print(f"AST-based (Time: {ast_time:.2f}ms):")
        if not ast_results:
            print("  No results.")
        else:
            for r in ast_results[:2]:
                print(f"  - {r['name']} ({r['type']}) in {r['file_path']}:{r['start_line']} (Score: {r['match_score']})")
        
        print(f"\nLSP-based (Time: {lsp_time:.2f}ms):")
        if not lsp_results:
            print("  No results.")
        else:
            for r in lsp_results[:2]:
                print(f"  - {r['name']} ({r['type']}) in {r['file_path']}:{r['start_line']} (Score: {r['match_score']})")
                
    lsp_engine.close()

if __name__ == "__main__":
    run_benchmark()
