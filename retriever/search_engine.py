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
        
        # Mapping for entities
        targets = [ent['word'].strip() for ent in ner_results if ent['entity_group'] in ['Variable', 'Function', 'Class', 'Data_Structure', 'Code_Block']]
        
        if verbose:
            print(f"Detected Intent: {intent}")
            print(f"Extracted Targets: {targets}")
            
        if not targets:
            return []

        # 3. Search AST Index with Fuzzy Matching
        results = []
        for target in targets:
            for item in self.ast_index:
                # Check match score
                score = fuzz.partial_ratio(target.lower(), item['name'].lower())
                
                if score >= fuzzy_threshold:
                    # Filter by intent if specific
                    if intent == "Find_Function" and item['type'] not in ['function', 'method']:
                        # Give some room for fuzzy but bias towards functions
                        if score < 90: continue 
                    if intent == "Find_Class" and item['type'] != 'class':
                        if score < 90: continue
                        
                    # Add score to item for ranking
                    item_copy = item.copy()
                    item_copy['match_score'] = score
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
