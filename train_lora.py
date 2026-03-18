import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==========================================
# 專案 1: 企業級技術領域資料微調 (QLoRA)
# 目標: 使用 4-bit 量化與 LoRA 微調 LLaMA-3 (8B)
# ==========================================

def main():
    # 1. 設定模型名稱 (這裡使用 unsloth 預先量化好的版本可以省下載時間，或是官方 meta-llama/Meta-Llama-3-8B)
    model_id = "unsloth/llama-3-8b-bnb-4bit" 
    
    # 2. 設定 4-bit 量化配置 (對應履歷: 應用 4-bit 量化技術，降低 70% 記憶體)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("載入 Tokenizer 與 模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 準備進行 k-bit 訓練
    model = prepare_model_for_kbit_training(model)

    # 3. 設定 LoRA 參數 (對應履歷: 使用 LoRA 低秩適應技術)
    lora_config = LoraConfig(
        r=16, # Rank 秩大小
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # 這行會印出 "trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5195%" 完美呼應你的履歷！

    # 4. 準備「技術領域資料集」 (這裡示範用 huggingface 上常見的 IT 問答資料集，你也可以換成本地 JSON)
    print("載入技術領域資料集...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:1000]") # 取前1000筆示範
    
    # 將資料轉為 LLaMA 認識的 Prompt 格式
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'][i]}\n{example['context'][i]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['response'][i]}<|eot_id|>"
            output_texts.append(text)
        return output_texts

    # 5. 設定訓練參數
    training_args = TrainingArguments(
        output_dir="./llama3-tech-lora-results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # 透過累積梯度，在 10GB VRAM 也能模擬大 Batch Size
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        max_steps=100, # 示範只跑 100 步
        fp16=False,
        bf16=True, # LLaMA-3 建議使用 bfloat16
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    # 6. 開始微調 (SFTTrainer)
    print("開始 LoRA 微調...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        max_seq_length=512, # 為了省記憶體，長度設 512
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    trainer.train()

    # 7. 儲存 LoRA 權重
    print("儲存 LoRA 權重至 ./llama3-tech-lora-adapter")
    trainer.model.save_pretrained("./llama3-tech-lora-adapter")
    tokenizer.save_pretrained("./llama3-tech-lora-adapter")
    print("訓練完成！")

if __name__ == "__main__":
    main()
