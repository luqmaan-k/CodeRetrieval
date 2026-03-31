import json
import torch
from pathlib import Path
from retriever.search_engine import SearchEngine
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs

def get_csn_index(lang_name, data):
    """
    Indexes CodeSearchNet functions specifically for evaluation.
    Each item in data is a CodeSearchNet record.
    """
    if lang_name == 'python':
        ts_lang = Language(tspython.language())
        scm_path = "retriever/grammars/tree-sitter-python/queries/tags.scm"
    elif lang_name == 'java':
        ts_lang = Language(tsjava.language())
        scm_path = "retriever/grammars/tree-sitter-java/queries/tags.scm"
    elif lang_name == 'javascript':
        ts_lang = Language(tsjs.language())
        scm_path = "retriever/grammars/tree-sitter-javascript/queries/tags.scm"
    else:
        return []

    parser = Parser(ts_lang)
    with open(scm_path, 'r') as f:
        query = Query(ts_lang, f.read())
    
    index = []
    cursor = QueryCursor(query)
    
    for idx, item in enumerate(data):
        code = item['func_code_string'].encode('utf8')
        tree = parser.parse(code)
        captures_dict = cursor.captures(tree.root_node)
        
        # We just need to know which function name belongs to which index for ground truth
        # CSN gives us the function name in 'func_name'
        # But we index all definitions found in that snippet
        for tag_name, nodes in captures_dict.items():
            if tag_name.startswith('definition.'):
                for node in nodes:
                    name = code[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                    index.append({
                        'name': name,
                        'type': tag_name.split('.')[-1],
                        'csn_idx': idx, # Link back to the original CSN record
                        'func_name': item['func_name'],
                        'file_path': f"csn_{idx}", # Dummy for deduplication
                        'start_line': 0
                    })
    return index

def evaluate():
    print("Starting Evaluation on CodeSearchNet Subsets...")
    
    engine = SearchEngine(
        intent_model_path="models/intent_classifier_final",
        ner_model_path="models/ner_model_final",
        ast_index_path="data/processed/ast_index.json" # Not used directly for CSN search
    )
    
    results_summary = {}
    
    for lang in ["python", "java", "javascript"]:
        data_path = Path(f"data/raw/csn_{lang}_test_subset.json")
        if not data_path.exists():
            continue
            
        print(f"\nEvaluating {lang}...")
        with open(data_path, 'r') as f:
            # CodeSearchNet JSONs are often line-delimited or a list. 
            # to_json(orient='records') usually makes a list in newer datasets.
            # Let's handle both.
            content = f.read()
            try:
                data = json.loads(content)
            except:
                data = [json.loads(l) for l in content.strip().split('\n')]
        
        # 1. Build a temporary index for this language's test set
        print(f"Indexing {len(data)} test snippets...")
        csn_index = get_csn_index(lang, data)
        engine.ast_index = csn_index # Override engine's index with CSN subset
        
        mrr = 0
        recall_1 = 0
        recall_5 = 0
        total_queries = 0
        
        for idx, item in enumerate(data):
            # Try both possible keys for documentation
            query_str = item.get('func_documentation_string') or item.get('func_docstring')
            if not query_str: continue
            
            total_queries += 1
            search_results = engine.query(query_str, top_k=5)
            
            # Ground truth: the result should have the correct csn_idx
            found_ranks = [i + 1 for i, res in enumerate(search_results) if res.get('csn_idx') == idx]
            
            if found_ranks:
                rank = found_ranks[0]
                mrr += 1.0 / rank
                if rank == 1: recall_1 += 1
                recall_5 += 1
        
        if total_queries > 0:
            stats = {
                "MRR": mrr / total_queries,
                "Recall@1": recall_1 / total_queries,
                "Recall@5": recall_5 / total_queries,
                "Total Queries": total_queries
            }
            results_summary[lang] = stats
            print(f"Results for {lang}:")
            for k, v in stats.items():
                print(f"  {k}: {v:.4f}")

    print("\n" + "="*30)
    print("Final Evaluation Summary")
    print("="*30)
    print(json.dumps(results_summary, indent=2))

if __name__ == "__main__":
    evaluate()
