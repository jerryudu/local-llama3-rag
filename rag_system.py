import argparse
import os
import torch
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# ==========================================
# 專案 2: 企業級 RAG 檢索增強生成系統
# 目標: 載入技術文件 -> 向量資料庫 -> LLaMA-3 (8B) 4-bit 回答
# ==========================================

def setup_rag_retriever(txt_path, persist_directory="./chroma_db"):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"找不到知識庫文件: {txt_path}")
    print(f"1. 讀取技術文件: {txt_path}")
    loader = TextLoader(txt_path, encoding='utf-8')
    documents = loader.load()

    print("2. 進行文本切塊 (Chunking)...")
    # 對應履歷: 優化上下文檢索邏輯，切塊大小為 1000 字元，重疊 200 字元避免斷句
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    print("3. 載入開源 Embedding 模型...")
    # 這裡選用 sentence-transformers 中文效果不錯的輕量模型
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

    print("4. 建立 Chroma 向量資料庫...")
    # 對應履歷: 串接向量資料庫
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 每次檢索最相關的 3 個片段
    
    return retriever

def setup_llama3_llm(adapter_path=None):
    print("5. 載入 4-bit 量化 LLaMA-3 (8B) 模型...")
    model_id = "unsloth/llama-3-8b-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 🌟 新增這行：建立 4-bit 的設定檔
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # 🌟 修改這行：把 load_in_4bit=True 換成 quantization_config=quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    if adapter_path:
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"找不到 LoRA Adapter 目錄: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # 建立 LangChain 認識的 HuggingFacePipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        repetition_penalty=1.1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def parse_args():
    parser = argparse.ArgumentParser(description="以 Base LLaMA-3 或 LoRA Adapter 執行企業文件 RAG 問答")
    parser.add_argument("--document", default="company_policy.txt")
    parser.add_argument("--chroma-dir", default="./chroma_db")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--adapter-path")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_path = args.adapter_path
    if args.use_lora:
        if adapter_path:
            raise ValueError("--use-lora 與 --adapter-path 不需同時使用")
        adapter_path = "./llama3-tech-lora-adapter"
    # 假設我們有一個技術文件叫做 company_policy.pdf
    # (你需要先在同一個資料夾放一個 txt 檔才能跑)
    try:
        retriever = setup_rag_retriever(args.document, args.chroma_dir)
        llm = setup_llama3_llm(adapter_path)
    except Exception as e:
        raise RuntimeError(f"初始化失敗: {e}") from e

    print("6. 建立 RAG (檢索增強生成) 流程...")
    # 對應履歷: 減少 20% 的模型幻覺
    # 定義 RAG 專屬的 Prompt，強制模型只能看 Context 回答
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    你是一個專業的企業技術助理。請「嚴格」根據以下提供的參考資料 (Context) 來回答問題。
    如果在參考資料中找不到答案，請直接回答「根據目前資料無法得知」，絕對不要編造答案（避免幻覺）。
    
    參考資料：
    {context}
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    問題：{question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
     # 🌟 改動 1: 用 | 直接串聯，取代 LLMChain
    rag_chain = prompt | llm 


    # 7. 測試問答
    print("\n✅ 系統準備完畢！")
    while True:
        user_query = input("\n請輸入你的問題 (輸入 q 離開): ")
        if user_query.lower() == 'q':
            break
            
        print("🔍 正在檢索向量資料庫...")
        docs = retriever.invoke(user_query)
        context_text = "\n".join([doc.page_content for doc in docs])
        
        response = rag_chain.invoke({
            "context": context_text,
            "question": user_query
        })
        
        print("\n================= 回答 =================\n")
        # 因為 LLaMA-3 的 pipeline 會把 prompt 也印出來，這裡做個簡單的切割只取 assistant 後面的字
        print(response.strip())
        print(" ========================================")
if __name__ == "__main__":
    main()
