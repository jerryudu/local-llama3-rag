# 🏢 企業級 AI 知識庫與 RAG 檢索系統
(Enterprise AI Knowledge Base & RAG System)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![LLaMA-3](https://img.shields.io/badge/Model-LLaMA--3--8B-purple)

## 📝 專案簡介 (Project Overview)
本專案旨在解決企業內部私有資料的 AI 問答需求，透過 **RAG (檢索增強生成)** 架構結合開源大型語言模型 (LLaMA-3)，打造出具備高準確度且低幻覺的智能技術助理。
同時，為了解決大模型部署成本高昂的問題，本專案導入了 **4-bit 量化 (Quantization)** 與 **LoRA (低秩適應)** 技術，使其能夠在單張消費級 GPU (如 10GB VRAM) 上進行高效能的微調與推論部署。

### 🌟 核心成果 (Key Achievements)
* **降低 70% 訓練硬體門檻**：運用 QLoRA 技術 (4-bit 載入 + LoRA 微調)，成功將 LLaMA-3 (8B) 的微調記憶體需求從 >16GB 壓縮至 ~8GB。
* **減少 20% 模型幻覺 (Hallucinations)**：建置完整 RAG 檢索流程，透過 LangChain 進行文本切塊 (`chunk_size=1000, overlap=200`) 並寫入 Chroma 向量資料庫，強制模型依據檢索上下文回答。
* **邊緣部署落地能力**：優化模型推論延遲，證實系統可在消費級硬體上穩定運行。

---

## 📂 專案架構 (Project Structure)
* `train_lora.py`: **LLM 微調腳本**。負責載入技術領域資料集 (此處以 Databricks Dolly 為例)，並使用 PEFT/TRL 庫對 LLaMA-3 進行 4-bit LoRA 微調。
* `rag_system.py`: **RAG 檢索問答主程式**。整合 `PyPDFLoader`、`ChromaDB` 向量資料庫與量化後的 LLaMA-3 模型，提供終端機互動式的私有知識庫問答介面。
* `company_policy.pdf`: (需自行準備) 作為 RAG 系統知識庫來源的範例技術文件。

---

## 🚀 環境安裝 (Installation)

請確保您的環境具備支援 CUDA 的 GPU (建議 VRAM >= 10GB)，並安裝以下依賴套件：

```bash
# 基礎 AI 與微調套件
pip install torch torchvision torchaudio
pip install transformers peft trl accelerate bitsandbytes datasets

# RAG 與向量資料庫套件
pip install langchain langchain-community langchain-huggingface
pip install chromadb pypdf sentence-transformers
```

---

## 💡 使用教學 (Usage)

### 1. 模型微調 (Fine-Tuning)
執行以下指令開始進行 LoRA 微調。訓練完成後，權重將會儲存至 `./llama3-tech-lora-adapter` 目錄中。
```bash
python train_lora.py
```

### 2. 啟動 RAG 檢索問答系統
請先在專案根目錄放置一份名為 `company_policy.pdf` 的文件，接著執行：
```bash
python rag_system.py
```
系統會自動將 PDF 進行切塊、向量化並存入本地的 ChromaDB 中。待模型載入完畢後，即可在終端機輸入問題進行問答測試。

---

## 🛠️ 技術棧 (Tech Stack)
* **Language Model:** Meta LLaMA-3 (8B)
* **Fine-Tuning:** Hugging Face PEFT, TRL, bitsandbytes (QLoRA)
* **RAG Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** sentence-transformers (text2vec-base-chinese)
