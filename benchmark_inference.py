import argparse
import json
import os
import statistics
import time


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def main():
    parser = argparse.ArgumentParser(description="測量 RAG inference latency")
    parser.add_argument("--document", default="company_policy.txt")
    parser.add_argument("--query", default="每週最多可以遠距工作幾天？")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--adapter-path")
    parser.add_argument("--output", default="results/inference_benchmark.json")
    args = parser.parse_args()
    from rag_system import setup_llama3_llm, setup_rag_retriever
    start = time.perf_counter(); llm = setup_llama3_llm(args.adapter_path); loading = time.perf_counter() - start
    retriever = setup_rag_retriever(args.document)
    def invoke():
        docs = retriever.invoke(args.query); context = "\n".join(d.page_content for d in docs); return llm.invoke(f"資料：{context}\n問題：{args.query}\n回答：")
    for _ in range(args.warmup): invoke()
    values = []
    for _ in range(args.runs):
        start = time.perf_counter(); invoke(); values.append(time.perf_counter() - start)
    result = {"model_loading_seconds": loading, "runs": args.runs, "warmup": args.warmup, "mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values), "p95_seconds": percentile(values, 95), "latencies_seconds": values}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file: json.dump(result, file, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
