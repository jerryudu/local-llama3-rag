import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ==========================================
# 專案 1: 企業級技術領域資料微調 (QLoRA)
# 目標: 使用 4-bit 量化與 LoRA 微調 LLaMA-3 (8B)
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="使用 QLoRA 微調 LLaMA-3")
    parser.add_argument("--model-id", default="unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--optim", default="adamw_torch", choices=["adamw_torch", "sgd"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./llama3-tech-lora-results")
    parser.add_argument("--adapter-output", default="./llama3-tech-lora-adapter")
    parser.add_argument("--metrics-output", default="results/training_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    # 1. 設定模型名稱 (這裡使用 unsloth 預先量化好的版本可以省下載時間，或是官方 meta-llama/Meta-Llama-3-8B)
    model_id = args.model_id
    
    # 2. 設定 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("載入 Tokenizer 與 模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model_load_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.cpu_offload:
        if not torch.cuda.is_available():
            raise RuntimeError("--cpu-offload 需要 CUDA 環境")
        model_load_kwargs["max_memory"] = {0: "6GiB", "cpu": "32GiB"}
        model_load_kwargs["offload_folder"] = os.path.join(args.output_dir, "offload")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_load_kwargs,
    )
    
    # 準備進行 k-bit 訓練
    model = prepare_model_for_kbit_training(model)

    # 3. 設定 LoRA 參數 (對應履歷: 使用 LoRA 低秩適應技術)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    # 4. 準備「技術領域資料集」 (這裡示範用 huggingface 上常見的 IT 問答資料集，你也可以換成本地 JSON)
    print("載入技術領域資料集...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split=f"train[:{args.dataset_size}]")
    
    # 將資料轉為 LLaMA 認識的 Prompt 格式
    def formatting_prompts_func(example):
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{example['instruction']}\n{example['context']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{example['response']}<|eot_id|>"
        )

    # 5. 設定訓練參數
    optimizer = "sgd" if args.cpu_offload and args.optim == "adamw_torch" else args.optim
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=optimizer,
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        max_steps=args.max_steps,
        fp16=False,
        bf16=True, # LLaMA-3 建議使用 bfloat16
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_length=args.max_seq_length,
    )
    model.config.use_cache = False

    # 6. 開始微調 (SFTTrainer)
    print("開始 LoRA 微調...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()

    # 7. 儲存 LoRA 權重
    print(f"儲存 LoRA 權重至 {args.adapter_output}")
    trainer.model.save_pretrained(args.adapter_output)
    tokenizer.save_pretrained(args.adapter_output)
    metrics = dict(train_result.metrics)
    metrics.update({
        "model_id": model_id,
        "dataset_size": args.dataset_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "optimizer": optimizer,
        "cpu_offload": args.cpu_offload,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameter_percent": trainable_parameters / total_parameters * 100,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
    })
    metrics_directory = os.path.dirname(args.metrics_output)
    if metrics_directory:
        os.makedirs(metrics_directory, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print("訓練完成！")

if __name__ == "__main__":
    main()
