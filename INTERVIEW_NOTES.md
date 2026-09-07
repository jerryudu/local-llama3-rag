# Interview Notes

## Project Goal
在有限 GPU VRAM 下展示 Local LLaMA-3 8B 結合 QLoRA 與 RAG，回答企業私有文件問題。

## Architecture
Dolly -> QLoRA -> LoRA Adapter; Base LLaMA-3 4-bit + Adapter <- Context <- Chroma <- Company Policy

## Key Points
4-bit quantization 降低載入成本，LoRA 只更新少量參數。企業政策知識主要由 RAG 提供，不是由 Dolly fine-tuning 寫入模型。

## Results
目前實測結果（RTX 3080、20 題 POC evaluation、LoRA Adapter）：No-RAG answer accuracy 0.0%、hallucination rate 20.0%；LoRA + RAG answer accuracy 40.0%、hallucination rate 0.0%、refusal accuracy 100.0%、overall success rate 60.0%。LoRA + RAG 共 8 題正確、8 題錯誤、4 題正確拒答。這次結果已排除 prompt/context 污染。

其他實測：QLoRA 100 steps 約 414 秒，peak allocated VRAM 約 9.56 GB，trainable parameters 0.4597%；4-bit model loading peak 約 5.31 GiB；修正 generation output 後的 LoRA inference median latency 約 1.926 秒，p95 約 1.984 秒。

## Limitations
Evaluation set 小、Dolly 不是企業資料、latency 依硬體而變化；本專案不包含 production authentication、RBAC 或 ACL。
