# Local LLaMA-3 RAG + QLoRA POC

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![LLaMA-3](https://img.shields.io/badge/Model-LLaMA--3--8B-purple)

## 📝 專案簡介 (Project Overview)
本專案旨在解決企業內部私有資料的 AI 問答需求，透過 **RAG (檢索增強生成)** 架構結合開源大型語言模型 (LLaMA-3)，打造出具備高準確度且低幻覺的智能技術助理。
同時，為了解決大模型部署成本高昂的問題，本專案導入了 **4-bit 量化 (Quantization)** 與 **LoRA (低秩適應)** 技術，使其能夠在單張消費級 GPU (如 10GB VRAM) 上進行高效能的微調與推論部署。

### Current Status
這是 Local LLaMA-3 8B + QLoRA + RAG 的 Proof of Concept。Accuracy、VRAM 與 latency 數字只以 `results/` 中的實測輸出為準。

---

## 📂 專案架構 (Project Structure)
* `train_lora.py`: **LLM 微調腳本**。負責載入技術領域資料集 (此處以 Databricks Dolly 為例)，並使用 PEFT/TRL 庫對 LLaMA-3 進行 4-bit LoRA 微調。
* `rag_system.py`: **RAG 檢索問答主程式**。使用 `TextLoader`、ChromaDB 與量化後的 LLaMA-3。
* `company_policy.txt`: RAG 系統使用的範例企業政策文件。
* `eval_questions.json`, `evaluate_rag.py`: 評估資料與 No-RAG / RAG 評估腳本。
* `benchmark_memory.py`, `benchmark_inference.py`: memory 與 latency benchmark。

---

## 🚀 環境安裝 (Installation)

請確保您的環境具備支援 CUDA 的 GPU (建議 VRAM >= 10GB)，並安裝以下依賴套件：

```bash
# 基礎 AI 與微調套件
pip install torch torchvision torchaudio
pip install transformers peft trl accelerate bitsandbytes datasets

# RAG 與向量資料庫套件
pip install langchain langchain-community langchain-huggingface
pip install langchain-text-splitters chromadb sentence-transformers
```

---

## 💡 使用教學 (Usage)

### 1. 模型微調 (Fine-Tuning)
執行以下指令開始進行 LoRA 微調。訓練完成後，權重將會儲存至 `./llama3-tech-lora-adapter` 目錄中。
```bash
python train_lora.py
```

### 2. 啟動 RAG 檢索問答系統
使用現有的 `company_policy.txt` 啟動 RAG：
```bash
python rag_system.py
python rag_system.py --use-lora
python rag_system.py --adapter-path ./llama3-tech-lora-adapter
```
系統會將 TXT 切塊、向量化並存入本地的 ChromaDB 中。

### Evaluation and Benchmarks

```bash
python evaluate_rag.py
python benchmark_memory.py
python benchmark_inference.py
python -m unittest discover -s tests -v
```

沒有 CUDA 時，memory benchmark 會停止且不會產生模擬結果。Dolly fine-tuning 是 instruction-following POC；企業政策知識主要由 RAG 提供。

### Measured Results

Results from the 20-question `company_policy.txt` evaluation on the Base model:

| Mode | Accuracy | Hallucination Rate | Refusal Accuracy |
| --- | ---: | ---: | ---: |
| No-RAG | 0.0% | 20.0% | 0.0% |
| RAG | 60.0% | 0.0% | 100.0% |

The RAG run produced 12 correct answers, 4 incorrect answers, and 4 correct refusals. These results are from a small POC set and are not a production quality or causal claim. The evaluation was run without the LoRA Adapter; LoRA + RAG requires a separate run with `--adapter-path`.

### Known Limitations

- Evaluation set 只有 20 題，只能代表小型 POC 結果。
- LoRA 是否提升 policy QA 必須由 evaluation 驗證。
- 4-bit latency 取決於 GPU、kernel 與 framework。
- 本專案未涵蓋 production authentication、RBAC、ACL 或 multi-user service。

---

## 🛠️ 技術棧 (Tech Stack)
* **Language Model:** Meta LLaMA-3 (8B)
* **Fine-Tuning:** Hugging Face PEFT, TRL, bitsandbytes (QLoRA)
* **RAG Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** sentence-transformers (text2vec-base-chinese)
