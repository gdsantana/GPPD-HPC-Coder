#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_with_adapters(base_model_name, adapter_path, device="cuda"):
    print(f"Loading base model: {base_model_name}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    print("✓ Model loaded successfully")
    return model, tokenizer


def load_merged_model(model_path, device="cuda"):
    print(f"Loading merged model from: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("✓ Model loaded successfully")
    return model, tokenizer


def generate_response(model, tokenizer, instruction, max_new_tokens=512, temperature=0.7, top_p=0.95):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response.split("### Response:\n")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer):
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        instruction = input("Enter instruction: ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not instruction:
            continue
        
        print("\nGenerating response...\n")
        
        response = generate_response(model, tokenizer, instruction)
        
        print("Response:")
        print("-" * 60)
        print(response)
        print("-" * 60 + "\n")


def batch_mode(model, tokenizer, instructions):
    print("\n" + "="*60)
    print("Batch Mode")
    print("="*60 + "\n")
    
    for i, instruction in enumerate(instructions, 1):
        print(f"[{i}/{len(instructions)}] Instruction: {instruction}")
        
        response = generate_response(model, tokenizer, instruction)
        
        print("Response:")
        print("-" * 60)
        print(response)
        print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned DeepSeek model")
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Path to LoRA adapters"
    )
    parser.add_argument(
        "--merged_model_path",
        type=str,
        help="Path to merged model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/deepseek-coder-6.7b-base",
        help="Base model name (only needed with --adapter_path)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Single instruction to generate response for"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling"
    )
    
    args = parser.parse_args()
    
    if not args.adapter_path and not args.merged_model_path:
        print("Error: Must specify either --adapter_path or --merged_model_path")
        parser.print_help()
        return
    
    if args.adapter_path:
        model, tokenizer = load_model_with_adapters(
            args.base_model,
            args.adapter_path
        )
    else:
        model, tokenizer = load_merged_model(args.merged_model_path)
    
    model.eval()
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    
    elif args.instruction:
        print(f"\nInstruction: {args.instruction}\n")
        response = generate_response(
            model,
            tokenizer,
            args.instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print("Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
    
    else:
        default_instructions = [
            "Write a CUDA kernel for vector addition",
            "Optimize this CUDA code for matrix multiplication",
            "Explain how shared memory works in CUDA",
        ]
        
        print("\nNo instruction provided. Running with default examples...")
        batch_mode(model, tokenizer, default_instructions)


if __name__ == "__main__":
    main()
