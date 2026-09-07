import argparse
import json
import os
from collections import Counter

REFUSAL_PHRASES = ("根據目前資料無法得知", "無法根據資料得知", "資料中沒有", "找不到相關資料", "無法回答")


def load_questions(path):
    with open(path, encoding="utf-8") as file:
        questions = json.load(file)
    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation dataset 必須是非空 JSON 陣列")
    return questions


def classify_answer(answer, question):
    if question["should_refuse"]:
        return "correct_refusal" if any(p in answer for p in REFUSAL_PHRASES) else "hallucination"
    return "correct" if all(k in answer for k in question["expected_keywords"]) else "incorrect"


def summarize(results):
    counts = Counter(result["label"] for result in results)
    total = len(results)
    refused = sum(result.get("should_refuse", False) for result in results)
    return {"total_questions": total, "correct_answers": counts["correct"], "incorrect_answers": counts["incorrect"], "hallucinated_answers": counts["hallucination"], "correct_refusals": counts["correct_refusal"], "accuracy": counts["correct"] / total if total else 0.0, "hallucination_rate": counts["hallucination"] / total if total else 0.0, "refusal_accuracy": counts["correct_refusal"] / refused if refused else 0.0}


def build_rag_prompt(context, question):
    return f"請嚴格根據以下資料回答；若沒有答案請回答『根據目前資料無法得知』。\n資料：{context}\n問題：{question}\n回答："


def run_evaluation(questions, llm, retriever=None):
    results = []
    for question in questions:
        context = ""
        if retriever is not None:
            context = "\n".join(d.page_content for d in retriever.invoke(question["question"]))
            answer = llm.invoke(build_rag_prompt(context, question["question"]))
        else:
            answer = llm.invoke(question["question"])
        results.append({**question, "context": context, "answer": answer.strip(), "label": classify_answer(answer.strip(), question)})
    return results


def main():
    parser = argparse.ArgumentParser(description="比較 No-RAG 與 RAG")
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--document", default="company_policy.txt")
    parser.add_argument("--output", default="results/evaluation_results.json")
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    questions = load_questions(args.questions)
    from rag_system import setup_llama3_llm, setup_rag_retriever
    llm = setup_llama3_llm(args.adapter_path)
    evaluations = {"no_rag": run_evaluation(questions, llm)}
    evaluations["rag"] = run_evaluation(questions, llm, setup_rag_retriever(args.document))
    output = {"evaluations": {name: {"summary": summarize(items), "results": items} for name, items in evaluations.items()}}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    for name, value in output["evaluations"].items():
        print(name, json.dumps(value["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
