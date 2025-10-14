#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Tenta carregar adapters PEFT (QLoRA/LoRA) se existirem
try:
    from peft import AutoPeftModelForCausalLM  # disponível em peft>=0.7
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Gera saídas a partir de prompts, usando modelo local (adapters ou full)."
    )
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Caminho local do modelo treinado (ex.: ./trained_model)."
    )
    p.add_argument(
        "--prompts_dir",
        type=str,
        required=True,
        help="Pasta com arquivos de prompt (serão lidos recursivamente)."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Pasta base para salvar as gerações."
    )
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Número de gerações por prompt."
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Máximo de tokens gerados por saída."
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura para amostragem."
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus sampling)."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente aleatória global (opcional)."
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Tentar carregar o modelo com quantização 4-bit (útil para QLoRA)."
    )
    return p.parse_args()


def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model_and_tokenizer(model_dir: str, load_in_4bit: bool):
    """
    Carrega AutoPeftModelForCausalLM (adapters LoRA/QLoRA) se possível.
    Se não, tenta AutoModelForCausalLM diretamente do diretorio.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    quant_cfg = None
    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if HAS_PEFT:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quant_cfg if load_in_4bit else None,
            )
            return model, tokenizer
        except Exception:
            # Fallback para modelo "full" (sem adapters) no mesmo diretório
            pass

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg if load_in_4bit else None,
    )
    return model, tokenizer


def read_prompt(fp: Path) -> str:
    return fp.read_text(encoding="utf-8")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.load_in_4bit)
    print(f"Modelo carregado de: {args.model_dir}")

    prompts_root = Path(args.prompts_dir)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    # Permite rodar em qualquer tokenizer que não tenha token de pad explícito
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        # como fallback, define pad_token = eos_token
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.eos_token_id

    # Busca recursiva por arquivos (ignora diretórios)
    prompt_files = [p for p in prompts_root.rglob("*") if p.is_file()]

    if not prompt_files:
        print(f"Nenhum arquivo de prompt encontrado em: {prompts_root}")
        return

    for fp in prompt_files:
        # Caminho espelhado dentro de output_dir
        rel = fp.relative_to(prompts_root)
        out_dir = out_root / rel.parent
        ensure_dir(out_dir)

        prompt = read_prompt(fp)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        for i in range(1, args.k + 1):
            # Se o usuário passou seed global, ainda assim variamos levemente
            # via torch.manual_seed diferente por amostra (opcional).
            if args.seed is not None:
                torch.manual_seed(args.seed + i)
                torch.cuda.manual_seed_all(args.seed + i)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=pad_id,
                )

            # Decodifica e, se o decodificado contiver o prompt no início, recorta-o
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if generated.startswith(prompt):
                completion = generated[len(prompt):].lstrip()
            else:
                completion = generated

            # Nome do arquivo de saída enumerado
            stem = fp.name
            out_name = f"{stem}.gen-{i:03d}.txt"
            out_path = out_dir / out_name
            out_path.write_text(completion, encoding="utf-8")
            print(f"[OK] {out_path}")

    print("Finalizado.")


if __name__ == "__main__":
    main()
