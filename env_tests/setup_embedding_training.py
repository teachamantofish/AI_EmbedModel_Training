# Setup script for embedding model fine-tuning on AMD hardware
# Run this before attempting to fine-tune your embedding model

"""
# 1. Set up the environment
cd C:\GIT\AI_RAG_pipeline_brogers\1Pipeline
python setup_embedding_training.py

# 2. Fine-tune the model (this will take time!)
python embedding_finetuner.py --action train

# 3. Test the fine-tuned model
python embedding_finetuner.py --action test

# 4. Export for Ollama/vector DB use
python embedding_finetuner.py --action export
"""

import subprocess
import sys
import platform
from pathlib import Path

def install_requirements():
    """Install required packages for embedding fine-tuning."""
    print("Installing Python requirements for embedding fine-tuning...")
    
    req_file = Path(__file__).parent / "requirements_embedding.txt"
    if not req_file.exists():
        print(f"Requirements file not found: {req_file}")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def check_pytorch_amd():
    """Check if PyTorch is configured for AMD GPU."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✓ CUDA available (NVIDIA GPU detected)")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ No CUDA devices detected")
        
        # Check for ROCm (AMD GPU support)
        if hasattr(torch, 'hip') and torch.hip.is_available():
            print("✓ ROCm/HIP available (AMD GPU detected)")
        else:
            print("ℹ No ROCm/HIP detected - will use CPU")
        
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def download_model_info():
    """Provide information about downloading the Qwen model."""
    print("\n" + "="*60)
    print("MODEL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
For Qwen3-Embedding-4B model:

Option 1: Automatic download (requires internet)
- The fine-tuning script will automatically download the model
- Requires ~8GB download and ~16GB storage space

Option 2: Manual download with Ollama
- Run: ollama pull qwen3-embedding:4b
- Or: ollama pull qwen2.5-coder:14b-instruct-q8_0 (for code tasks)

Option 3: Use a smaller model for testing
- The script will fallback to all-MiniLM-L6-v2 (~90MB)
- Good for testing the pipeline before using larger models

For your AMD hardware:
- 64GB RAM: Can handle the 4B model easily
- iGPU: Use batch_size=8-16 depending on allocated GPU memory
- NPU: Consider smaller quantized models for better performance
""")

def check_hardware():
    """Check system hardware for fine-tuning readiness."""
    print("\n" + "="*60)
    print("HARDWARE CHECK")
    print("="*60)
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"System RAM: {ram_gb:.1f} GB")
        
        if ram_gb >= 32:
            print("✓ Sufficient RAM for large models")
        elif ram_gb >= 16:
            print("⚠ Adequate RAM - consider smaller models or lower batch sizes")
        else:
            print("✗ Low RAM - use small models and batch_size=1")
            
    except ImportError:
        print("Install psutil to check RAM: pip install psutil")
    
    # Check OS
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # Check if running on Windows (for AMD GPU support)
    if platform.system() == "Windows":
        print("✓ Windows detected - good for AMD GPU support")
    else:
        print("ℹ Non-Windows OS - check ROCm compatibility")

def main():
    print("Setting up embedding model fine-tuning environment...")
    print("="*60)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please install requirements manually.")
        return
    
    # Check PyTorch
    if not check_pytorch_amd():
        print("PyTorch setup issue. You may need to install PyTorch manually.")
        print("For AMD GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6")
    
    # Check hardware
    check_hardware()
    
    # Model download info
    download_model_info()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Ensure your training data is ready:
   python embedmodel_tuner.py

2. Start fine-tuning:
   python embedding_finetuner.py --action train

3. Test the fine-tuned model:
   python embedding_finetuner.py --action test

4. Export for Ollama/local use:
   python embedding_finetuner.py --action export

For AMD hardware optimization, edit config/embedding_training_config.py
""")

if __name__ == "__main__":
    main()