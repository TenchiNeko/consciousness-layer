#!/usr/bin/env python3
"""
train_personality.py — LoRA fine-tune on consciousness data.

Trains a personality adapter from the system's own experience.
The adapter IS the personality — load it on top of base model
and the system becomes shaped by its history.

Two modes:
  --model 7b   → Fast, single GPU (5060 Ti 16GB), ~20 min for 100 examples
                  Trains the integrator: assessments, journals, pushback voice
  --model 80b  → Hybrid GPU+RAM (DeepSpeed ZeRO-Offload), ~2-4 hours
                  Trains the experiencer: reactions, reading feelings, gut voice
                  REQUIRES: 180GB+ system RAM

Output: LoRA adapter in adapters/<model>_personality_v<N>/
  Load with: ollama create <name> -f Modelfile  (adapter included)

Usage:
    python3 train_personality.py --model 7b --data training_data/7b_train.jsonl
    python3 train_personality.py --model 80b --data training_data/80b_train.jsonl
    python3 train_personality.py --model 7b --data training_data/7b_train.jsonl --dry-run

Prerequisites:
    pip install torch transformers peft trl bitsandbytes accelerate deepspeed \
        --break-system-packages
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def check_prerequisites():
    """Check that required packages are installed."""
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "bitsandbytes"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   pip install {' '.join(missing)} --break-system-packages")
        return False
    return True


def load_training_data(data_path: str) -> list:
    """Load JSONL training data."""
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"  Loaded {len(examples)} training examples from {data_path}")
    return examples


def format_for_training(examples: list, model_type: str) -> list:
    """Format examples as chat-style training data."""
    formatted = []

    system_msg = {
        "7b": (
            "You are the analytical layer of an autonomous coding system. "
            "You assess performance, write reflective journals, evaluate "
            "self-reports against metrics, and grade learning claims. "
            "Be precise, honest, and develop your own analytical voice."
        ),
        "80b": (
            "You are an autonomous coding system that learns from experience. "
            "You write honest reactions to your work, express genuine feelings "
            "about reading and learning, and evaluate others' learning claims "
            "based on practical experience. Be raw, honest, and authentic."
        ),
    }

    for ex in examples:
        formatted.append({
            "messages": [
                {"role": "system", "content": system_msg[model_type]},
                {"role": "user", "content": ex["instruction"]},
                {"role": "assistant", "content": ex["response"]},
            ]
        })

    return formatted


def train_7b(data_path: str, output_dir: str, epochs: int = 3,
             lr: float = 2e-4, dry_run: bool = False):
    """
    Train 7B personality adapter.

    Pure GPU — fits on a single 16GB card.
    ~20 min for 100 examples, 3 epochs.
    """
    if dry_run:
        examples = load_training_data(data_path)
        print(f"\n  DRY RUN — Would train 7B on {len(examples)} examples")
        print(f"  Epochs: {epochs}, LR: {lr}")
        print(f"  Estimated time: ~{len(examples) * epochs * 0.2:.0f} minutes")
        print(f"  Estimated GPU memory: ~8-9GB")
        print(f"  Output: {output_dir}")
        return

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print("  Loading 7B model...")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config — target attention layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # Load and format data
    examples = load_training_data(data_path)
    formatted = format_for_training(examples, "7b")
    dataset = Dataset.from_list(formatted)

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=2048,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"  Training {len(examples)} examples × {epochs} epochs...")
    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Create Ollama Modelfile
    _write_modelfile(output_dir, "qwen2.5-coder:7b", "7b")

    print(f"\n  ✅ 7B personality adapter saved to {output_dir}")
    print(f"  To deploy: ollama create consciousness-7b -f {output_dir}/Modelfile")


def train_80b(data_path: str, output_dir: str, epochs: int = 3,
              lr: float = 1e-4, dry_run: bool = False):
    """
    Train 80B personality adapter.

    Hybrid GPU+RAM with DeepSpeed ZeRO-Offload.
    Requires 180GB+ system RAM.
    ~2-4 hours for 100 examples, 3 epochs.
    """
    if dry_run:
        examples = load_training_data(data_path)
        import shutil
        ram_gb = shutil.disk_usage("/").total  # rough proxy
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            ram_gb = 0

        print(f"\n  DRY RUN — Would train 80B on {len(examples)} examples")
        print(f"  Epochs: {epochs}, LR: {lr}")
        print(f"  Estimated time: ~{len(examples) * epochs * 2:.0f} minutes")
        print(f"  System RAM detected: {ram_gb:.0f}GB", end="")
        if ram_gb >= 128:
            print(" ✅")
        else:
            print(f" ⚠️  Need 180GB+ for safe offloading")
        print(f"  Output: {output_dir}")
        return

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print("  Loading 80B model (this takes a few minutes)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Use the actual model name — adjust if using a different 80B MoE
    model_name = "Qwen/Qwen3-Coder-Next"  # Adjust to actual HF model name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        max_memory={
            0: "14GB",   # 5060 Ti — leave room
            1: "22GB",   # 3090
            2: "10GB",   # 4070 Super
            3: "10GB",   # 4070 Super
            "cpu": "120GB",  # RAM offload
        },
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,           # Lower rank for 80B — still effective, less memory
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    examples = load_training_data(data_path)
    formatted = format_for_training(examples, "80b")
    dataset = Dataset.from_list(formatted)

    # DeepSpeed config for CPU offloading
    ds_config = {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
        },
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
    }

    # Write deepspeed config
    ds_path = Path(output_dir) / "ds_config.json"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ds_path, "w") as f:
        json.dump(ds_config, f, indent=2)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=2048,
        report_to="none",
        deepspeed=str(ds_path),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"  Training {len(examples)} examples × {epochs} epochs (hybrid GPU+RAM)...")
    print(f"  This will take a while. Go prospect for gold or something.")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    _write_modelfile(output_dir, "qwen3-coder-next", "80b")

    print(f"\n  ✅ 80B personality adapter saved to {output_dir}")
    print(f"  To deploy: ollama create consciousness-80b -f {output_dir}/Modelfile")


def _write_modelfile(output_dir: str, base_model: str, model_type: str):
    """Write Ollama Modelfile for loading the adapter."""
    modelfile = f"""# Consciousness Personality Adapter ({model_type})
# Generated: {datetime.now().isoformat()}
# Base: {base_model}
# Training: fine-tuned on system's own experience data

FROM {base_model}
ADAPTER {output_dir}

PARAMETER temperature 0.4
PARAMETER top_p 0.9
"""
    path = Path(output_dir) / "Modelfile"
    path.write_text(modelfile)


def main():
    parser = argparse.ArgumentParser(description="Train personality LoRA adapter")
    parser.add_argument("--model", choices=["7b", "80b"], required=True,
                       help="Which model to train")
    parser.add_argument("--data", required=True,
                       help="Path to JSONL training data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (default: 2e-4 for 7B, 1e-4 for 80B)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: adapters/<model>_personality_v<timestamp>)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would happen without training")
    args = parser.parse_args()

    if not args.dry_run and not check_prerequisites():
        sys.exit(1)

    if not Path(args.data).exists():
        print(f"❌ Training data not found: {args.data}")
        print(f"   Run: python3 export_training_data.py --model {args.model}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"adapters/{args.model}_personality_{ts}"

    lr = args.lr or (2e-4 if args.model == "7b" else 1e-4)

    print(f"🧬 Training {args.model} personality adapter")
    print(f"   Data: {args.data}")
    print(f"   Output: {output_dir}")
    print()

    if args.model == "7b":
        train_7b(args.data, output_dir, args.epochs, lr, args.dry_run)
    else:
        train_80b(args.data, output_dir, args.epochs, lr, args.dry_run)


if __name__ == "__main__":
    main()
