import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
from thefuzz import fuzz

class SearchEngine:
    def __init__(self, intent_model_path, ner_model_path, ast_index_path):
        print("Loading Search Engine components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Intent Model
        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path).to(self.device)
        self.intent_pipe = pipeline("text-classification", model=self.intent_model, tokenizer=self.intent_tokenizer, device=self.device)
        
        # Load NER Model
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path).to(self.device)
        self.ner_pipe = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, device=self.device, aggregation_strategy="simple")
        
        # Load AST Index
        with open(ast_index_path, 'r') as f:
            self.ast_index = json.load(f)
        
        print(f"Search Engine ready. Indexed {len(self.ast_index)} definitions.")

    def query(self, user_input, top_k=5, fuzzy_threshold=70, verbose=False):
        # Manually truncate input to stay safely within 512 tokens (leaving room for special tokens)
        tokens = self.intent_tokenizer.encode(user_input, truncation=True, max_length=510, add_special_tokens=False)
        user_input = self.intent_tokenizer.decode(tokens)
        
        # 1. Classify Intent
        intent_results = self.intent_pipe(user_input)
        intent = intent_results[0]['label']
        
        # 2. Extract Entities
        ner_results = self.ner_pipe(user_input)
        if verbose: print(f"Raw NER results: {ner_results}")
        
        # Mapping for entities - combine contiguous entities if they seem to belong to the same name
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
                        # HEURISTIC: If it's the ONLY entity or the LAST entity in a "find file" context, it might be a target
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
        
        # If we have targets AND file_filters, and the target is a substring of file_filter, 
        # it might be a mis-extraction (e.g. "search_engine.py" -> "search" (Function), "engine.py" (File_Name))
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
        if not targets and len(user_input.split()) <= 6:
             stops = {'find', 'show', 'where', 'is', 'the', 'class', 'function', 'method', 'defined', 'me', 'in'}
             fallback = [w for w in user_input.lower().split() if w not in stops]
             if fallback:
                 targets.extend(fallback)

        if verbose:
            print(f"Detected Intent: {intent}")
            print(f"Extracted Targets: {targets}")
            print(f"File Filters: {file_filters}")
            
        if not targets:
            return []

        # 3. Search AST Index with Fuzzy Matching
        results = []
        for target in targets:
            for item in self.ast_index:
                # Apply file filter if present
                if file_filters:
                    matches_file = any(fuzz.partial_ratio(f.lower(), item['file_path'].lower()) >= 90 for f in file_filters)
                    if not matches_file:
                        continue

                # Check match score
                # Use ratio for better whole-word matching if target is long enough
                if len(target) > 3:
                    score = fuzz.ratio(target.lower(), item['name'].lower())
                    # Also try partial if ratio is low
                    if score < 80:
                        score = max(score, fuzz.partial_ratio(target.lower(), item['name'].lower()))
                else:
                    score = fuzz.ratio(target.lower(), item['name'].lower())
                
                if score >= fuzzy_threshold:
                    # Filter by intent if specific
                    is_func = item['type'] in ['function', 'method']
                    is_class = item['type'] in ['class', 'interface', 'struct']
                    
                    if intent == "Find_Function" and not is_func:
                        if score < 95: continue 
                    if intent == "Find_Class" and not is_class:
                        if score < 95: continue
                        
                    # Calculate final ranking score
                    final_score = score
                    if target.lower() == item['name'].lower():
                        final_score += 20 # Perfect match bonus
                    
                    # Boost results that match the intent more specifically
                    if intent == "Find_Function" and is_func:
                        final_score += 10
                    if intent == "Find_Class" and is_class:
                        final_score += 10
                        
                    # Add score to item for ranking
                    item_copy = item.copy()
                    item_copy['match_score'] = final_score
                    results.append(item_copy)
        
        # Sort by score and remove duplicates
        results = sorted(results, key=lambda x: x['match_score'], reverse=True)
        
        seen = set()
        unique_results = []
        for r in results:
            key = (r['file_path'], r['start_line'])
            if key not in seen:
                unique_results.append(r)
                seen.add(key)
                
        return unique_results[:top_k]

if __name__ == "__main__":
    # Test the engine
    engine = SearchEngine(
        intent_model_path="models/intent_classifier_final",
        ner_model_path="models/ner_model_final",
        ast_index_path="data/processed/ast_index.json"
    )
    
    test_queries = [
        "Find the function calculateSum",
        "Where is the Calculator class defined?",
        "Show me the multiply method"
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        res = engine.query(q)
        print(json.dumps(res, indent=2))
