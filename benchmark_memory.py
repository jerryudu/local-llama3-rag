import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="測量 4-bit 模型 GPU memory")
    parser.add_argument("--model-id", default="unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument("--output", default="results/memory_benchmark.json")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，未產生模擬結果。")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
    result = {"model_id": args.model_id, "gpu_name": torch.cuda.get_device_name(0), "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory, "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved()}
    del model
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
