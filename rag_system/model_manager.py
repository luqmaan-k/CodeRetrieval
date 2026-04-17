from huggingface_hub import hf_hub_download
import os

def download_qwen():
    repo_id = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
    filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    local_dir = "models"
    
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    
    if os.path.exists(local_path):
        print(f"Model already exists at {local_path}")
        return local_path
        
    print(f"Downloading {filename} from {repo_id}...")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print(f"Downloaded to {path}")
    return path

if __name__ == "__main__":
    download_qwen()
