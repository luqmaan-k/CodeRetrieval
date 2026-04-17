import json
import torch
import time
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
from thefuzz import fuzz
from .lsp_client import LSPClient, uri_to_path, SYMBOL_KIND_MAP

class LSPSearchEngine:
    def __init__(self, intent_model_path, ner_model_path, root_dir=".", suppress_output=False):
        if suppress_output:
            import transformers
            transformers.utils.logging.set_verbosity_error()
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            print("Loading Search Engine components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.root_dir = Path(root_dir).absolute()
        
        # Load NLP Models
        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path).to(self.device)
        self.intent_pipe = pipeline("text-classification", model=self.intent_model, tokenizer=self.intent_tokenizer, device=self.device)
        
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path).to(self.device)
        self.ner_pipe = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, device=self.device, aggregation_strategy="simple")
        
        # Initialize LSP Client (Python for now)
        self.lsp_client = LSPClient(["jedi-language-server"], self.root_dir.as_uri())
        if not suppress_output: print("Initializing LSP Client...")
        self.lsp_client.initialize()
        if not suppress_output: print("LSP Client ready. Waiting 2s for indexing...")
        time.sleep(2)
        
    def query(self, user_input, top_k=5, fuzzy_threshold=70, verbose=False):
        # 0. Truncate input
        tokens = self.intent_tokenizer.encode(user_input, truncation=True, max_length=510, add_special_tokens=False)
        user_input_truncated = self.intent_tokenizer.decode(tokens)
        
        # 1. Classify Intent
        intent_results = self.intent_pipe(user_input_truncated)
        intent = intent_results[0]['label']
        
        # Heuristic for Usages/References
        is_usage_query = any(kw in user_input.lower() for kw in ['use', 'usage', 'reference', 'call', 'where is it used'])
        if is_usage_query:
            intent = "Find_Usages"
            
        # 2. Extract Entities
        ner_results = self.ner_pipe(user_input_truncated)
        
        targets = []
        file_filters = []
        if ner_results:
            current_target = ""
            current_group = ""
            for i, ent in enumerate(ner_results):
                group = ent['entity_group']
                word = ent['word'].strip()
                
                # If it starts with ## or is immediately adjacent, combine it
                if current_target and (ent['word'].startswith('##') or ent['start'] == ner_results[i-1]['end']):
                    current_target += word.lstrip('#')
                else:
                    if current_target:
                        if current_group == 'File_Name':
                            file_filters.append(current_target)
                        elif current_group in ['Variable', 'Function', 'Class', 'Data_Structure', 'Code_Block', 'Application']:
                            targets.append(current_target)
                    current_target = word.lstrip('#')
                    current_group = group
            
            if current_target:
                if current_group == 'File_Name':
                    file_filters.append(current_target)
                elif current_group in ['Variable', 'Function', 'Class', 'Data_Structure', 'Code_Block', 'Application']:
                    targets.append(current_target)
        
        # Refine: if we have both, keep as is. If we ONLY have File_Name, it's the target.
        if not targets and file_filters:
            targets = [f.replace('.py', '').replace('.java', '').replace('.js', '').replace('.cpp', '').replace('.cs', '') for f in file_filters]
            file_filters = []
        
        # If we have targets AND file_filters, and the target is a substring of file_filter
        if targets and file_filters:
            new_targets = []
            for t in targets:
                is_part_of_file = any(t in f for f in file_filters)
                if not is_part_of_file:
                    new_targets.append(t)
            if not new_targets: # Everything was part of a file name
                targets = file_filters
                file_filters = []
            else:
                targets = new_targets

        # If still no targets, try simple heuristic
        if not targets and len(user_input_truncated.split()) <= 6:
             stops = {'find', 'show', 'where', 'is', 'the', 'class', 'function', 'method', 'defined', 'me', 'in'}
             fallback = [w for w in user_input_truncated.lower().split() if w not in stops]
             if fallback:
                 targets.extend(fallback)

        if verbose:
            print(f"Intent: {intent}, Targets: {targets}, Filters: {file_filters}")
            
        if not targets: return []

        # 3. Search LSP Workspace Symbols
        all_symbols = []
        for target in targets:
            res = self.lsp_client.workspace_symbol(target)
            if res and 'result' in res and res['result'] is not None:
                # Apply file filter if present
                if file_filters:
                    filtered = []
                    for sym in res['result']:
                        uri = sym['location']['uri'] if 'location' in sym else sym['locationUri']
                        path = uri_to_path(uri)
                        if any(fuzz.partial_ratio(f.lower(), path.lower()) >= 90 for f in file_filters):
                            filtered.append(sym)
                    all_symbols.extend(filtered)
                else:
                    all_symbols.extend(res['result'])
            else:
                if verbose: print(f"LSP failed to find symbols for target '{target}'")
                
        if intent == "Find_Usages":
            # For each found symbol, find its references
            all_refs = []
            for sym in all_symbols[:3]: # Limit to top 3 matching symbols to avoid explosion
                uri = sym['location']['uri'] if 'location' in sym else sym['locationUri']
                range_info = sym['location']['range'] if 'location' in sym else sym['range']
                line = range_info['start']['line']
                char = range_info['start']['character']
                
                ref_res = self.lsp_client.references(uri, line, char)
                if ref_res and 'result' in ref_res and ref_res['result'] is not None:
                    for ref in ref_res['result']:
                        # Add metadata to reference
                        ref['symbol_name'] = sym['name']
                        all_refs.append(ref)
            
            # Map refs back to results format
            results = []
            for ref in all_refs:
                uri = ref['uri']
                path = uri_to_path(uri)
                if not path.startswith(str(self.root_dir)): continue
                if any(part.startswith('.') or part in {'venv', 'bak_venv', '__pycache__', 'node_modules'} for part in Path(path).parts):
                    continue
                
                results.append({
                    'name': f"Usage of {ref['symbol_name']}",
                    'type': 'usage',
                    'file_path': os.path.relpath(path, self.root_dir),
                    'start_line': ref['range']['start']['line'] + 1,
                    'end_line': ref['range']['end']['line'] + 1,
                    'match_score': 100 # High score for explicit usage search
                })
            return results[:top_k]

        # 4. Filter and Rank
        results = []
        for sym in all_symbols:
            name = sym['name']
            kind_id = sym['kind']
            kind_name = SYMBOL_KIND_MAP.get(kind_id, "Unknown")
            uri = sym['location']['uri'] if 'location' in sym else sym['locationUri']
            path = uri_to_path(uri)
            
            # Skip symbols outside root_dir or in ignored dirs
            if not path.startswith(str(self.root_dir)):
                continue
            if any(part.startswith('.') or part in {'venv', 'bak_venv', '__pycache__', 'node_modules'} for part in Path(path).parts):
                continue

            # Fuzzy match score
            score = 0
            for target in targets:
                if len(target) > 3:
                    s = fuzz.ratio(target.lower(), name.lower())
                    if s < 80:
                        s = max(s, fuzz.partial_ratio(target.lower(), name.lower()))
                else:
                    s = fuzz.ratio(target.lower(), name.lower())
                score = max(score, s)
            
            if score < fuzzy_threshold:
                continue
                
            # Intent matching
            is_func = kind_name in ["Function", "Method"]
            is_class = kind_name in ["Class", "Interface", "Struct"]
            
            # Strict Intent Filtering (matching AST engine)
            if intent == "Find_Function" and not is_func:
                if score < 95: continue 
            if intent == "Find_Class" and not is_class:
                if score < 95: continue
            
            # Perfect match bonus
            if any(target.lower() == name.lower() for target in targets):
                score += 20
                
            # Boost results that match the intent more specifically
            if intent == "Find_Function" and is_func:
                score += 10
            if intent == "Find_Class" and is_class:
                score += 10
                
            range_info = sym['location']['range'] if 'location' in sym else sym['range']
            
            results.append({
                'name': name,
                'type': kind_name.lower(),
                'file_path': os.path.relpath(path, self.root_dir),
                'start_line': range_info['start']['line'] + 1,
                'end_line': range_info['end']['line'] + 1,
                'match_score': score
            })

        # Sort and deduplicate
        results = sorted(results, key=lambda x: x['match_score'], reverse=True)
        seen = set()
        unique_results = []
        for r in results:
            key = (r['file_path'], r['start_line'])
            if key not in seen:
                unique_results.append(r)
                seen.add(key)
                
        return unique_results[:top_k]

    def close(self):
        self.lsp_client.shutdown()

if __name__ == "__main__":
    engine = LSPSearchEngine(
        intent_model_path="models/intent_classifier_final",
        ner_model_path="models/ner_model_final"
    )
    
    test_queries = [
        "Find the SearchEngine class",
        "Where is SearchEngine used?",
        "Show me the fetch_data function"
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        res = engine.query(q, verbose=True)
        print(json.dumps(res, indent=2))
        
    engine.close()
