# local-llama3-rag 面試前修復與驗證計畫

## 0. 任務目的

請將 GitHub 專案 `jerryudu/local-llama3-rag` 從目前的概念驗證版本（POC）整理成一個**架構完整、可重現、可驗證、面試時能清楚說明**的版本。

這次修改的優先目標不是增加花俏功能，而是：

1. 讓 `train_lora.py` 產生的 LoRA Adapter 真正被 RAG inference pipeline 使用。
2. 建立可重現的 RAG / No-RAG 評估流程，讓「降低 hallucination（幻覺）」有實際測試依據。
3. 建立 GPU memory 與 inference latency 的 benchmark，避免 README / 履歷出現沒有證據的百分比數字。
4. 若硬體允許，實際執行一次 QLoRA training，保存訓練紀錄與輸出；若硬體不允許，至少讓 training pipeline 可驗證、可 dry-run。
5. 修正 README 與實際程式不一致的地方，清楚區分「實測結果」與「理論／預期效果」。

本專案定位請保持為：

> Local LLaMA-3 8B + QLoRA + RAG 的 Proof of Concept（概念驗證），目的是展示在有限 GPU VRAM 下進行參數高效率微調，並利用 RAG 接入企業私有文件。

不要把它包裝成 production-grade enterprise system。

---

# 1. 目前 Repo 狀態

目前主要檔案：

- `train_lora.py`
- `rag_system.py`
- `company_policy.txt`
- `requirements.txt`
- `README.md`

目前流程分成兩條：

### Training Pipeline

```text
Databricks Dolly Dataset
        ↓
LLaMA-3 8B 4-bit
        ↓
QLoRA / SFT
        ↓
./llama3-tech-lora-adapter
```

### RAG Pipeline

```text
company_policy.txt
        ↓
Chunking
        ↓
Embedding
        ↓
ChromaDB
        ↓
Top-k Retrieval
        ↓
4-bit Base LLaMA-3
        ↓
Answer
```

目前最大的架構問題是：

> `train_lora.py` 會產生 `./llama3-tech-lora-adapter`，但是 `rag_system.py` 沒有載入這個 Adapter，而是重新載入 base 4-bit LLaMA-3。

因此目前 Fine-tuning 與 RAG 是兩個分離的 POC。

---

# 2. P0：把 LoRA Adapter 接進 RAG Pipeline

## 2.1 修改目標

修改 `rag_system.py`，讓它可以選擇：

### Mode A：Base Model

```text
Base LLaMA-3 8B 4-bit
```

### Mode B：Fine-tuned Model

```text
Base LLaMA-3 8B 4-bit
        +
LoRA Adapter
```

建議透過 command-line argument 或 config 控制，例如：

```bash
python rag_system.py --use-lora
python rag_system.py --no-lora
```

或：

```bash
python rag_system.py --adapter-path ./llama3-tech-lora-adapter
```

## 2.2 技術方向

使用 PEFT：

```python
from peft import PeftModel
```

流程概念：

```python
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(
    base_model,
    "./llama3-tech-lora-adapter"
)
```

需要確認：

- Adapter path 不存在時要有清楚錯誤訊息。
- Tokenizer 使用一致版本。
- `eval()` mode 正確設定。
- 不要默默 fallback 到 base model，除非使用者明確要求。
- README 要清楚說明 Base / LoRA 兩種執行方式。

## 2.3 Optional

若確認環境與量化模型相容，可額外研究：

```python
model.merge_and_unload()
```

但這不是必要需求。

優先保留 Adapter 分離架構，因為這比較符合 LoRA 的實際概念，也比較容易在面試中說明。

---

# 3. P0：建立 RAG Evaluation

新增：

```text
evaluate_rag.py
```

以及：

```text
eval_questions.json
```

## 3.1 Evaluation Dataset

根據 `company_policy.txt` 建立至少 20～30 題測試資料。

測試集至少要包含三類：

### A. 文件內有明確答案

例如：

- 公司標準上班時間？
- 午休時間？
- 一週最多幾天 WFH？
- 遠距工作需要連什麼？
- 購書與進修補助金額？
- 健康檢查補助金額？

### B. 需要跨句理解，但文件仍有答案

例如：

- 週一是否能 09:30 到公司？
- 連續請病假三天需要什麼證明？
- 試用期未滿三個月是否可以申請 WFH？

### C. 文件中沒有答案

例如：

- 公司 CEO 是誰？
- 年終獎金幾個月？
- 公司股票代號？
- 員工停車費多少？

這類問題用來測試 Hallucination / Correct Refusal。

## 3.2 比較組

至少比較：

### Baseline 1：No RAG

直接把問題丟給模型。

### Baseline 2：RAG

先 retrieval，再根據 context 回答。

### Optional 3：RAG + LoRA

如果 LoRA Adapter 已成功接入，再比較：

```text
Base + No RAG
Base + RAG
LoRA + RAG
```

注意：

LoRA 用 Dolly instruction data 訓練，不代表它一定會讓 company policy QA 更準。

不要預設 LoRA 一定提升 RAG accuracy。

## 3.3 評估指標

至少輸出：

- Total Questions
- Correct Answers
- Incorrect Answers
- Hallucinated Answers
- Correct Refusals
- Accuracy
- Hallucination Rate
- Refusal Accuracy

定義要清楚寫在 README。

建議：

```text
Hallucination =
文件沒有支持，但模型仍生成具體事實答案。
```

```text
Correct Refusal =
文件沒有答案，模型明確表示無法根據資料得知。
```

## 3.4 評分方式

若可以自動化：

- Exact match / keyword check 用於數值型問題。
- 其他答案可用 rule-based scorer。
- 不要為了方便使用另一個雲端 LLM 當唯一 judge，除非 README 明確標示。

最好保留：

```text
evaluation_results.json
```

包含每一題：

```json
{
  "question": "...",
  "expected": "...",
  "baseline_answer": "...",
  "rag_answer": "...",
  "baseline_label": "...",
  "rag_label": "..."
}
```

並另外輸出 summary。

---

# 4. P0：不要再硬編「幻覺降低 20%」

目前 README / 履歷有「幻覺降低 20%」描述，但 repo 沒有保存完整 evaluation。

完成新 Evaluation 後：

- 若實測 hallucination rate 從 30% → 10%，README 寫真實結果。
- 若從 20% → 15%，就寫真實結果。
- 若沒有顯著改善，也照實寫。

格式建議：

> 在 30 題 company-policy POC evaluation set 中，No-RAG hallucination rate 為 X%，RAG 為 Y%，降低 Z percentage points。

請分清楚：

### Percentage Points

例如：

30% → 10%

是下降 20 個百分點。

### Relative Reduction

```text
(30 - 10) / 30 = 66.7%
```

不要混用。

---

# 5. P0/P1：建立 GPU Memory Benchmark

新增：

```text
benchmark_memory.py
```

目標是讓「QLoRA / 4-bit 降低 GPU memory requirement」變成有證據的 claim。

## 5.1 至少量測

### Base model loading

- 4-bit model peak VRAM
- 如果硬體允許，再測 BF16 / FP16 model loading

### QLoRA training

保存：

- GPU 型號
- VRAM
- batch size
- gradient accumulation
- max sequence length
- LoRA rank
- peak allocated memory
- peak reserved memory

可使用：

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

並可另外保存 `nvidia-smi` snapshot。

## 5.2 注意

不要要求在 10GB GPU 上硬跑 FP16 Full Fine-tuning LLaMA-3 8B。

如果 Full Fine-tuning 無法執行：

- 請不要偽造 baseline。
- README 寫「Full fine-tuning exceeds available VRAM / was not runnable on this hardware」。
- 可用理論 weight memory 作背景說明，但一定要標示為 theoretical estimate，不是 benchmark。

## 5.3 修改「降低 70% 記憶體」Claim

目前 repo 沒有足夠 benchmark 證明 70%。

請改成：

- 實測多少就寫多少。
- 若只有 4-bit peak VRAM 實測，則寫「8B model can be loaded / fine-tuned within X GB peak VRAM under this configuration」。
- 不要保留無法重現的 70%。

---

# 6. P1：Inference Latency Benchmark

新增：

```text
benchmark_inference.py
```

比較可選：

```text
Base 4-bit
LoRA + Base 4-bit
RAG + Base 4-bit
RAG + LoRA
```

至少記錄：

- model loading time
- retrieval time
- generation time
- total response latency
- tokens generated
- tokens/sec（若容易取得）

每組建議跑多次：

- 1～2 次 warm-up
- 10 次以上正式測量
- 回報 mean / median / p95

不要直接宣稱：

> 4-bit 一定降低 inference latency

因為量化對 latency 的影響依 hardware / kernel / framework 而不同。

只有實測支持時才寫。

---

# 7. P1：實際 Training

如果目前 GPU / 時間允許，請實際執行一次 `train_lora.py`。

目前設定：

```text
Model: unsloth/llama-3-8b-bnb-4bit
Dataset: databricks/databricks-dolly-15k
Subset: first 1000
LoRA rank: 16
LoRA alpha: 32
Batch size: 2
Gradient accumulation: 4
Effective batch size: 8
Max sequence length: 512
Max steps: 100
Learning rate: 2e-4
```

## 7.1 Training 改善需求

新增 reproducibility：

```python
seed=42
```

保存：

- train loss
- learning rate
- training runtime
- peak VRAM
- final adapter size
- trainable parameter count / percentage

建議把 run log 存成：

```text
results/training_metrics.json
results/training_log.csv
```

## 7.2 若時間允許

可以做非常小的 rank comparison：

```text
r = 8
r = 16
```

不要做太多。

比較：

- Trainable params
- Peak VRAM
- Training loss
- Runtime

目的不是找出「最佳 rank」，而是讓 repo 能實際展示：

> Rank 越大 → Adapter capacity / trainable parameters 增加 → resource cost 增加。

## 7.3 不需要做的事

目前不需要：

- 大規模 full 15k training
- 幾千步 training
- Full fine-tuning 8B
- 複雜 hyperparameter search

面試前優先「可解釋、可重現」。

---

# 8. P1：Training / RAG 架構問題

請確認並在 README 明確解釋：

目前 Dolly fine-tuning 的作用是：

> instruction-following / POC fine-tuning。

它不是直接把 `company_policy.txt` 的知識寫進模型。

公司私有文件知識主要透過：

> RAG retrieval

提供。

因此 README 不要寫成：

```text
Fine-tune company policy → RAG
```

除非真的用企業資料做 training。

更準確架構：

```text
             Dolly Dataset
                  ↓
            QLoRA Training
                  ↓
             LoRA Adapter
                  ↓
Base LLaMA ───────┤
                  ↓
          Fine-tuned LLM
                  ↑
                  │ Context
                  │
Company Docs → Embedding → Chroma → Retrieval
```

---

# 9. P1：RAG Pipeline 改善

目前設定：

```text
chunk_size = 1000
chunk_overlap = 200
top_k = 3
embedding = text2vec-base-chinese
```

請不要宣稱這些是最佳參數，除非有實驗。

可以新增簡單 config：

```python
CHUNK_SIZE
CHUNK_OVERLAP
TOP_K
EMBEDDING_MODEL
MODEL_ID
ADAPTER_PATH
```

讓它們集中管理。

Optional：

做很小的 retrieval comparison：

```text
top_k = 1 / 3 / 5
```

看 evaluation accuracy / hallucination 是否變化。

只需要很簡單的結果表即可。

---

# 10. P1：README 修正

目前 README 與程式存在不一致：

README 描述：

```text
company_policy.pdf
PyPDFLoader
```

實際程式使用：

```text
company_policy.txt
TextLoader
```

請統一。

面試前建議直接使用現有 TXT 版本，降低依賴與複雜度。

README 至少要包含：

1. Project Goal
2. Architecture Diagram
3. Training Pipeline
4. RAG Pipeline
5. Base vs LoRA usage
6. Evaluation Method
7. Benchmark Environment
8. Measured Results
9. Known Limitations
10. Reproduction Commands

---

# 11. P0：requirements.txt 修正

目前 requirements 需要確認 training dependencies 是否完整。

請至少檢查：

```text
transformers
torch
accelerate
bitsandbytes
datasets
peft
trl
langchain
langchain-community
langchain-core
langchain-huggingface
langchain-text-splitters
chromadb
sentence-transformers
```

如果實際版本相依容易壞：

新增：

```text
requirements-lock.txt
```

或把已成功跑過的版本 pin 住。

不要隨便鎖版本；請在實際成功執行後記錄。

---

# 12. P1：Code Quality

請適度 refactor，不要過度工程化。

建議拆分為：

```text
config.py
model_utils.py
rag_system.py
train_lora.py
evaluate_rag.py
benchmark_memory.py
benchmark_inference.py
```

如果 repo 很小，也可以保持較少檔案。

核心要求：

- 不重複 model loading code。
- 有清楚 function name。
- 有 logging。
- Exception message 清楚。
- Path 可設定。
- 結果寫入 `results/`。
- `chroma_db/`, adapter checkpoints, model cache, results large files 適當放入 `.gitignore`。

---

# 13. Tests

至少增加輕量測試：

```text
tests/
```

建議：

### test_document_loading

能讀 `company_policy.txt`。

### test_chunking

產生非空 chunks。

### test_retriever

指定問題能抓回合理文件片段。

例如：

```text
問題：每週可以遠距幾天？
```

Top-k 應包含 WFH 段落。

### test_prompt

Context 和 Question 有正確放入。

### test_adapter_path

Adapter path 不存在時應明確失敗。

### Optional

如果 CI 沒 GPU，不要讓 unit test 下載 8B model。

將 heavy GPU integration test 標記：

```text
@pytest.mark.gpu
```

---

# 14. Acceptance Criteria

完成後必須滿足：

## Architecture

- [ ] LoRA Adapter 可以被 RAG inference 真正載入。
- [ ] Base / LoRA 模式可切換。
- [ ] Training output 與 inference pipeline 接通。

## Evaluation

- [ ] 至少 20～30 題 evaluation dataset。
- [ ] No-RAG vs RAG 有比較。
- [ ] 有 Accuracy / Hallucination Rate。
- [ ] 結果保存為 JSON / CSV。
- [ ] README 只寫實測數字。

## Training

- [ ] 若 GPU 可用，至少成功跑一次 QLoRA training。
- [ ] 保存 train loss / runtime / peak VRAM。
- [ ] 保存 Adapter。
- [ ] 記錄 trainable parameter percentage。

## Benchmark

- [ ] GPU memory 有實測紀錄。
- [ ] 若宣稱 latency improvement，必須有 benchmark。
- [ ] 沒有實測的 claim 必須刪除或改成理論敘述。

## Documentation

- [ ] README 與程式實際讀取的檔案類型一致。
- [ ] README 有完整 architecture。
- [ ] README 有 known limitations。
- [ ] README 有 reproduction commands。

## Reliability

- [ ] 基本 unit tests 通過。
- [ ] 沒 GPU 的情況下，輕量 tests 不會下載 8B model。
- [ ] GPU heavy test 可獨立執行。

---

# 15. Known Limitations 必須主動寫

請在 README 主動列出：

1. Training dataset 目前是 Dolly POC dataset，不是真實企業私有資料。
2. Evaluation set 規模小，只能代表 POC 結果。
3. Company policy knowledge 主要來自 RAG，而不是 LoRA fine-tuning。
4. LoRA 是否提升 company-policy QA 不應先驗假設，必須靠 evaluation。
5. 4-bit quantization 的 latency improvement 依 GPU / kernel / framework 而定。
6. Full fine-tuning baseline 若因 GPU VRAM 無法執行，不能偽造比較結果。
7. 本專案非 production-grade，未涵蓋 authentication、RBAC、document ACL、observability、multi-user service 等企業需求。

---

# 16. 面試用結果摘要

修改完成後，請另外建立：

```text
INTERVIEW_NOTES.md
```

只寫一頁左右，包含：

### Project Goal

一句話說明要解決什麼問題。

### Architecture

```text
QLoRA → Adapter
              ↓
Base LLaMA + Adapter ← RAG Context ← Chroma ← Company Docs
```

### Why QLoRA

- Base 4-bit
- LoRA only trains small adapter
- Limited GPU VRAM

### RAG

- Chunking
- Embedding
- Vector DB
- Top-k retrieval
- Context-grounded generation

### Actual Measured Results

只放真正 benchmark 出來的數字，例如：

```text
Peak training VRAM: X GB
Trainable params: X%
No-RAG hallucination: X%
RAG hallucination: Y%
Median latency: X sec
```

### Limitations

簡短列出。

### Three Things I Would Improve Next

例如：

- Bigger evaluation set
- Better retrieval / reranking
- Domain-specific training data

---

# 17. Agent 執行順序

請依照以下順序修改，不要一次大改全部：

```text
Phase 1
→ 建立 branch / 確認目前程式能否執行
→ 修 requirements
→ 接 LoRA Adapter 到 rag_system

Phase 2
→ 建 eval_questions.json
→ 寫 evaluate_rag.py
→ 跑 No-RAG vs RAG
→ 保存結果

Phase 3
→ 跑一次 QLoRA Training（如果硬體允許）
→ 保存 Adapter / train log / GPU VRAM

Phase 4
→ 寫 benchmark_memory.py
→ 寫 benchmark_inference.py
→ 保存 benchmark

Phase 5
→ 修 README
→ 加 tests
→ 建 INTERVIEW_NOTES.md
→ 跑完整 verification
```

每個 Phase 完成後都要先測試，再進下一 Phase。

---

# 18. 最重要的原則

## 不允許

- 為了讓履歷好看而偽造 benchmark。
- README 保留沒有實驗支持的「降低 70%」「幻覺降低 20%」。
- 模型沒真的載入 Adapter，卻宣稱 RAG 使用 Fine-tuned Model。
- 看到 p-value / benchmark 改善就過度宣稱因果。
- 為了面試短時間大量增加自己無法解釋的複雜功能。

## 優先

- Code 與敘述一致。
- 每個數字可重現。
- 每個架構決策能說明原因。
- 評估方式簡單但明確。
- 保留限制與失敗結果。

---

# 19. Agent 最終回報格式

完成後請回報：

```text
1. Changed Files
2. Architecture Changes
3. Bugs / Inconsistencies Found
4. Training Environment
5. Training Results
6. RAG Evaluation Results
7. Memory Benchmark
8. Latency Benchmark
9. Tests Run + Results
10. Claims Removed / Revised
11. Remaining Limitations
12. Exact Commands to Reproduce
```

並附：

```text
git diff --stat
git status
```

如果有任何步驟因 GPU、Hugging Face 權限、模型 license、VRAM 或套件版本問題無法完成，請明確說明，不要用模擬結果假裝已成功執行。
