import os
import sys
from pathlib import Path
from retriever.search_engine import SearchEngine
from rag_system.main import LangChainCodeRetriever, get_rag_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def highlight(text, color="94"): # default blue
    return f"\033[{color}m{text}\033[0m"

def print_retrieval_debug(query, engine):
    """Manually run the engine once to show the debug info the user wants."""
    print(f"\n{highlight('--- RETRIEVAL DEBUG INFO ---', '95')}")
    results = engine.query(query, verbose=True) # verbose=True shows Intent and NER
    
    if not results:
        print(f"{highlight('NO CONTEXT FOUND', '91')}")
    else:
        print(f"\n{highlight('RETRIEVED SNIPPETS:', '93')}")
        for i, res in enumerate(results):
            print(f"\n[{i+1}] {highlight(res['name'], '96')} ({res['type']})")
            print(f"File: {res['file_path']} | Line: {res['start_line']}")
            
            # Show a small preview
            try:
                full_path = Path(res['file_path'])
                if not full_path.is_absolute():
                    full_path = Path(".").absolute() / full_path
                with open(full_path, 'r') as f:
                    lines = f.readlines()
                    start = res['start_line'] - 1
                    preview = "".join(lines[start:start+3]).strip()
                    print(f"Code:\n{preview}\n...")
            except:
                pass
    print(highlight("-" * 28, "95"))

def run_rag_demo():
    print("\n" + "="*60)
    print(highlight("RAG System: AI Codebase Expert (AST Mode)", "96"))
    print("Asking Qwen2.5-Coder about the project using Tree-sitter context.")
    print("="*60)

    intent_model = "models/intent_classifier_final"
    ner_model = "models/ner_model_final"
    ast_index = "data/processed/ast_index.json"
    
    if not Path(intent_model).exists():
        print("Error: Models not found.")
        return

    # Initialize AST Engine
    engine = SearchEngine(intent_model, ner_model, ast_index)
    
    try:
        print("\nLoading LLM (this may take a few seconds)...")
        # 1. Show Retrieval Debug
        query = "Explain how the LSPClient communicates with the language server."
        print(f"\n{highlight('DEMO QUESTION:', '93')} {query}")
        
        print_retrieval_debug(query, engine)
        
        # 2. Get LLM Response
        rag_chain = get_rag_chain(engine)
        print("\nLLM is generating explanation...")
        response = rag_chain.invoke(query)
        
        print(f"\n{highlight('AI RESPONSE:', '92')}")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        print(f"\n{highlight('Interactive Mode:', '95')} (Type 'exit' to quit)")
        while True:
            user_q = input("\nAsk about the code > ").strip()
            if user_q.lower() in ['exit', 'quit']:
                break
            if not user_q: continue
            
            # Show debug for interactive too
            print_retrieval_debug(user_q, engine)
            
            print("\nThinking...")
            res = rag_chain.invoke(user_q)
            print(f"\n{highlight('AI RESPONSE:', '92')}\n{res}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    run_rag_demo()
