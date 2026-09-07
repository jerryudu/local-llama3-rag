import unittest
from pathlib import Path
from evaluate_rag import build_rag_prompt, classify_answer, load_questions, summarize


class EvaluationTests(unittest.TestCase):
    def test_dataset(self):
        questions = load_questions(Path(__file__).parents[1] / "eval_questions.json")
        self.assertGreaterEqual(len(questions), 20)
        self.assertTrue(any(q["should_refuse"] for q in questions))

    def test_answer_labels(self):
        question = {"expected_keywords": ["5,000 元"], "should_refuse": False}
        self.assertEqual(classify_answer("補助是 5,000 元", question), "correct")
        refusal = {"expected_keywords": [], "should_refuse": True}
        self.assertEqual(classify_answer("根據目前資料無法得知", refusal), "correct_refusal")
        self.assertEqual(classify_answer("CEO 是王小明", refusal), "hallucination")

    def test_context_keywords_do_not_affect_answer_label(self):
        question = {"expected_keywords": ["2 天"], "should_refuse": False}
        prompt = build_rag_prompt("WFH 每週最多 2 天。", "每週可以 WFH 幾天？")
        self.assertIn("2 天", prompt)
        self.assertEqual(classify_answer("每週可以 WFH 5 天。", question), "incorrect")

    def test_prompt_refusal_phrase_does_not_affect_answer_label(self):
        question = {"expected_keywords": [], "should_refuse": True}
        prompt = build_rag_prompt("沒有 CEO 資訊。", "公司 CEO 是誰？")
        self.assertIn("根據目前資料無法得知", prompt)
        self.assertEqual(classify_answer("CEO 是王小明。", question), "hallucination")

    def test_summary(self):
        summary = summarize([{"label": "correct", "should_refuse": False}, {"label": "hallucination", "should_refuse": True}])
        self.assertEqual(summary["hallucination_rate"], 0.5)
        self.assertEqual(summary["answer_accuracy"], 0.5)
        self.assertEqual(summary["overall_success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
