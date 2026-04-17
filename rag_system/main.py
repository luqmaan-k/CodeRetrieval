import os
from typing import List, Optional, Union
from pathlib import Path

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import LlamaCpp

from lsp_retriever.search_engine import LSPSearchEngine
from retriever.search_engine import SearchEngine

class LangChainCodeRetriever(BaseRetriever):
    engine: Union[SearchEngine, LSPSearchEngine]
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.engine.query(query, top_k=self.top_k)
        docs = []
        for res in results:
            # Try to read the code snippet
            content = f"File: {res['file_path']}\nLine: {res['start_line']}\n"
            try:
                full_path = Path(res['file_path'])
                if not full_path.is_absolute():
                    full_path = Path(".").absolute() / full_path
                with open(full_path, 'r') as f:
                    lines = f.readlines()
                    start = res['start_line'] - 1
                    end = min(res['end_line'] + 10, len(lines)) # Give some context
                    snippet = "".join(lines[start:end])
                    content += f"Code:\n{snippet}"
            except Exception as e:
                content += f"Code: (Could not read: {e})"
            
            docs.append(Document(page_content=content, metadata=res))
        return docs

def get_rag_chain(engine: Union[SearchEngine, LSPSearchEngine]):
    # 1. Setup Retriever
    retriever = LangChainCodeRetriever(engine=engine)

    # 2. Setup LLM (Qwen2.5-Coder via LlamaCpp)
    model_path = "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    if not os.path.exists(model_path):
        # Check if it's in the current dir
        if os.path.exists("qwen2.5-coder-7b-instruct-q4_k_m.gguf"):
            model_path = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        else:
            print(f"Model not found at {model_path}. Please run download_model.py first.")
            raise FileNotFoundError(model_path)
    
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=33, # Adjust based on 16GB VRAM
        verbose=False,
    )

    # 3. Setup Prompt
    template = """You are a senior software engineer. Use the following retrieved code snippets to answer the user's question about the codebase.
If you don't know the answer based on the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Build Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
