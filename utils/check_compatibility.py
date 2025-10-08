#!/usr/bin/env python3
import sys
import torch


def check_cuda():
    print("🔍 Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        print("   Please install PyTorch with CUDA support")
        print("   Visit: https://pytorch.org/get-started/locally/")
        return False
    
    print(f"✓ CUDA available")
    print(f"  Version: {torch.version.cuda}")
    return True


def check_gpu():
    print("\n🎮 Checking GPU...")
    if not torch.cuda.is_available():
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✓ GPU: {gpu_name}")
    print(f"  Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 20:
        print(f"⚠️  WARNING: GPU has less than 20GB memory")
        print(f"   Recommended: Use smaller max_length and lora_r")
        return False
    
    if "4090" not in gpu_name and "3090" not in gpu_name and "A100" not in gpu_name:
        print(f"⚠️  GPU is not RTX 4090/3090 or A100")
        print(f"   Fine-tuning may be slower or require more optimization")
    
    return True


def check_bf16():
    print("\n🔢 Checking BF16 support...")
    if not torch.cuda.is_available():
        return False
    
    if torch.cuda.is_bf16_supported():
        print("✓ BF16 supported - will use for faster training")
        return True
    else:
        print("⚠️  BF16 not supported - will use FP16 instead")
        return False


def check_packages():
    print("\n📦 Checking required packages...")
    
    packages = [
        ("transformers", "4.36.0"),
        ("peft", "0.7.0"),
        ("trl", "0.7.4"),
        ("bitsandbytes", "0.41.0"),
        ("accelerate", "0.25.0"),
        ("datasets", "2.14.0"),
    ]
    
    all_ok = True
    
    for package, min_version in packages:
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: not installed")
            all_ok = False
    
    print("\n🌟 Checking optional packages...")
    
    try:
        import flash_attn
        print(f"✓ flash-attn: {flash_attn.__version__}")
        print("  Recommendation: Use --use_flash_attention flag")
    except ImportError:
        print("⚠️  flash-attn: not installed")
        print("  Install with: pip install flash-attn --no-build-isolation")
        print("  Training will work without it but will be slower")
    
    return all_ok


def estimate_memory(max_length=1024, lora_r=64, batch_size=1):
    print(f"\n💾 Memory estimation for DeepSeek-Coder-6.7B...")
    print(f"  Parameters: max_length={max_length}, lora_r={lora_r}, batch_size={batch_size}")
    
    base_model_4bit = 4.5
    
    lora_memory = (lora_r * 0.01)
    
    activation_memory = (max_length * batch_size * 0.002)
    
    optimizer_memory = (lora_r * 0.02)
    
    overhead = 2.0
    
    total = base_model_4bit + lora_memory + activation_memory + optimizer_memory + overhead
    
    print(f"\n  Estimated breakdown:")
    print(f"    Model (4-bit):     {base_model_4bit:.1f} GB")
    print(f"    LoRA adapters:     {lora_memory:.1f} GB")
    print(f"    Activations:       {activation_memory:.1f} GB")
    print(f"    Optimizer states:  {optimizer_memory:.1f} GB")
    print(f"    Overhead:          {overhead:.1f} GB")
    print(f"  ─────────────────────────")
    print(f"  Total estimate:      {total:.1f} GB")
    
    if torch.cuda.is_available():
        available = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n  Available GPU memory: {available:.1f} GB")
        
        if total > available * 0.95:
            print(f"  ⚠️  Configuration may cause OOM!")
            print(f"     Try: --max_length 512 --lora_r 32")
        elif total > available * 0.85:
            print(f"  ⚠️  Configuration is tight, may be unstable")
            print(f"     Consider: --max_length 768 --lora_r 48")
        else:
            print(f"  ✓ Configuration should fit comfortably")
    
    return total


def suggest_config():
    if not torch.cuda.is_available():
        print("\n❌ Cannot suggest config without GPU")
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\n💡 Recommended configurations for {gpu_memory:.0f}GB GPU:")
    print("="*60)
    
    if gpu_memory >= 24:
        print("\n🚀 Balanced (Recommended):")
        print("  python finetune_deepseek_optimized.py \\")
        print("    --max_length 1024 \\")
        print("    --lora_r 64 \\")
        print("    --lora_alpha 128 \\")
        print("    --gradient_accumulation_steps 16 \\")
        print("    --use_flash_attention \\")
        print("    --packing")
        
        print("\n🎯 High Quality:")
        print("  python finetune_deepseek_optimized.py \\")
        print("    --max_length 2048 \\")
        print("    --lora_r 128 \\")
        print("    --lora_alpha 256 \\")
        print("    --gradient_accumulation_steps 32 \\")
        print("    --use_flash_attention \\")
        print("    --packing")
    
    elif gpu_memory >= 16:
        print("\n⚡ Conservative (Recommended):")
        print("  python finetune_deepseek_optimized.py \\")
        print("    --max_length 512 \\")
        print("    --lora_r 32 \\")
        print("    --lora_alpha 64 \\")
        print("    --gradient_accumulation_steps 16")
        
        print("\n🔧 Moderate:")
        print("  python finetune_deepseek_optimized.py \\")
        print("    --max_length 768 \\")
        print("    --lora_r 48 \\")
        print("    --lora_alpha 96 \\")
        print("    --gradient_accumulation_steps 24")
    
    else:
        print(f"\n⚠️  {gpu_memory:.0f}GB may not be sufficient for DeepSeek-6.7B")
        print("   Consider using a smaller model like deepseek-coder-1.3b-base")
    
    print("\n📊 Quick test command (100 samples):")
    print("  python finetune_deepseek_optimized.py \\")
    print("    --max_samples 100 \\")
    print("    --epochs 1 \\")
    print("    --save_steps 50")


def main():
    print("="*60)
    print("DeepSeek-Coder Fine-tuning Compatibility Check")
    print("="*60)
    
    cuda_ok = check_cuda()
    if not cuda_ok:
        print("\n❌ CUDA check failed - cannot proceed")
        sys.exit(1)
    
    gpu_ok = check_gpu()
    bf16_ok = check_bf16()
    packages_ok = check_packages()
    
    if not packages_ok:
        print("\n❌ Some packages are missing")
        print("   Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    estimate_memory(max_length=1024, lora_r=64, batch_size=1)
    suggest_config()
    
    print("\n" + "="*60)
    if cuda_ok and gpu_ok and packages_ok:
        print("✅ System ready for fine-tuning!")
    else:
        print("⚠️  Some issues detected - review warnings above")
    print("="*60)


if __name__ == "__main__":
    main()
