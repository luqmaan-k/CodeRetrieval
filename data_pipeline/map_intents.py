import json
import pandas as pd
from pathlib import Path

def map_intents():
    print("Mapping intents for CoSQA and CoNaLa...")
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "conala": raw_dir / "conala_train.json",
        "cosqa": raw_dir / "cosqa_train.json"
    }
    
    mapped_data = []
    
    # Intent mapping rules (simple keyword-based for bootstrapping)
    def get_intent(text):
        text = text.lower()
        if any(kw in text for kw in ['function', 'method', 'def', 'call', 'how to use', 'write a function']):
            return "Find_Function"
        elif any(kw in text for kw in ['class', 'object', 'instance', 'constructor']):
            return "Find_Class"
        elif any(kw in text for kw in ['what does', 'explain', 'understand', 'how it works']):
            return "Explain_Code"
        else:
            return "Find_General_Logic"

    # Process CoNaLa
    if datasets["conala"].exists():
        print("Processing CoNaLa...")
        with open(datasets["conala"], 'r') as f:
            for line in f:
                item = json.loads(line)
                query = item.get('rewritten_intent') or item.get('intent')
                if query:
                    mapped_data.append({
                        'query': query,
                        'intent': get_intent(query),
                        'source': 'conala'
                    })

    # Process CoSQA
    if datasets["cosqa"].exists():
        print("Processing CoSQA...")
        with open(datasets["cosqa"], 'r') as f:
            for line in f:
                item = json.loads(line)
                query = item.get('doc') # In CoSQA, 'doc' is the NL query
                if query:
                    mapped_data.append({
                        'query': query,
                        'intent': get_intent(query),
                        'source': 'cosqa'
                    })

    df = pd.DataFrame(mapped_data)
    output_path = processed_dir / "mapped_intents.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Mapped {len(df)} queries to {output_path}")
    print("\nIntent distribution:")
    print(df['intent'].value_counts())

if __name__ == "__main__":
    map_intents()
