from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Typed schemas for structured grading
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer matches the reference"]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the answer addresses the question"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    grounded: Annotated[bool, ..., "True if the answer is supported by the docs"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Reasoning for the score"]
    relevant: Annotated[bool, ..., "True if retrieved docs relate to the question"]

# Grader prompts
correctness_instructions = """
    You are grading a QA task.
    You get a QUESTION, a REFERENCE (ground truth) ANSWER, and a MODEL ANSWER.
    Score True only if the MODEL ANSWER is factually consistent with the REFERENCE.
    It's fine if the model adds extra details as long as they don't contradict the reference.
    Explain briefly, then output the boolean.
"""
relevance_instructions = """
    You are grading if a MODEL ANSWER addresses a QUESTION.
    Score True if the answer is concise and helps answer the question. Explain briefly, then output the boolean.
"""
grounded_instructions = """
    You are grading if a MODEL ANSWER is supported by the provided FACTS (retrieved docs).
    Score True if the answer stays within the facts and does not hallucinate. Explain briefly, then output the boolean.
"""
retrieval_relevance_instructions = """
    You are grading if the retrieved FACTS are relevant to the QUESTION.
    Score True if the facts contain any keywords or semantic meaning related to the question, even if some content is extra. Explain briefly, then output the boolean.
"""

# Create graders
def _make_llm_structured(schema):
    base = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    return base.with_structured_output(schema, method="json_schema", strict=True)

_grader_correctness = _make_llm_structured(CorrectnessGrade)
_grader_relevance = _make_llm_structured(RelevanceGrade)
_grader_grounded = _make_llm_structured(GroundedGrade)
_grader_retrieval = _make_llm_structured(RetrievalRelevanceGrade)

def eval_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    payload = f"QUESTION: {inputs['question']}\nREFERENCE ANSWER: {reference_outputs['answer']}\nMODEL ANSWER: {outputs['answer']}"
    grade = _grader_correctness.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": payload},
    ])
    return bool(grade["correct"])

def eval_relevance(inputs: dict, outputs: dict) -> bool:
    payload = f"QUESTION: {inputs['question']}\nMODEL ANSWER: {outputs['answer']}"
    grade = _grader_relevance.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": payload},
    ])
    return bool(grade["relevant"])

def eval_groundedness(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    payload = f"FACTS:\n{doc_string}\n\nMODEL ANSWER: {outputs['answer']}"
    grade = _grader_grounded.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": payload},
    ])
    return bool(grade["grounded"])

def eval_retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    docs = outputs.get("documents") or []
    doc_string = "\n\n".join(getattr(d, "page_content", "") for d in docs)
    payload = f"FACTS:\n{doc_string}\n\nQUESTION: {inputs['question']}"
    grade = _grader_retrieval.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": payload},
    ])
    return bool(grade["relevant"])

# Default benchmark dataset for evaluation
def build_sample_dataset():
    return [
        {
            "inputs": {"question": "What were the sales for the iPad in 2025?"},
            "outputs": {"answer": "The iPad net sales for the three months ended June 28, 2025, were $6,581 million, and for the nine months ended June 28, 2025, were $21,071 million."},
        },
        {
            "inputs": {"question": "What was Tesla's gross profit in 2025 compared to 2024?"},
            "outputs": {"answer": "Tesla's total gross profit in Q3 2025 was $5,054 million, which is a 1% increase compared to $4,997 million in Q3 2024."},
        },
        {
            "inputs": {"question": "What are the total net sales of Apple in 2025 so far?"},
            "outputs": {"answer": "The total net sales of Apple in 2025 so far (for the nine months ended June 28, 2025) are $313,695 million."},
        },
        {
            "inputs": {"question": "What was the operating income of Apple in Japan in Q3 of 2025?"},
            "outputs": {"answer": "The operating income of Apple in Japan for Q3 of 2025 (three months ended June 28, 2025) was $2,872 million."},
        },
        {
            "inputs": {"question": "What are the basic and diluted earnings per share for Apple so far in 2025?"},
            "outputs": {"answer": "For the nine months ended June 28, 2025, Apple's basic earnings per share were $5.64, and the diluted earnings per share were $5.62."},
        },
        {
            "inputs": {"question": "What was the driving factor of a change in net sales for Apple in Europe in 2025?"},
            "outputs": {"answer": "The increase in net sales for Apple in Europe during 2025 was primarily due to higher net sales of Services and iPhone."},
        },
        {
            "inputs": {"question": "Which Apple product has contributed the most to sales in 2025?"},
            "outputs": {"answer": "The iPhone contributed the most to Apple's sales in 2025, with net sales of $44,582 million for the third quarter and $160,561 million for the nine months ended June 28, 2025."}
        }
    ]

def run_evaluation(generate_fn, dataset=None):
    """
    generate_fn: callable(str) -> {'answer': str, 'documents': List[Document]}
    dataset: optional list of {'inputs': {'question'}, 'outputs': {'answer'}}
    """
    dataset = dataset or build_sample_dataset()
    results = []
    
    print("[EVAL] Running RAG evaluation on", len(dataset), "examples")
    for i, ex in enumerate(dataset, 1):
        q = ex["inputs"]["question"]
        ref = ex["outputs"]
        out = generate_fn(q)
        correctness = eval_correctness(ex["inputs"], out, ref)
        grounded = eval_groundedness(ex["inputs"], out)
        relevance = eval_relevance(ex["inputs"], out)
        retr_rel = eval_retrieval_relevance(ex["inputs"], out)
        results.append({
            "question": q,
            "answer": out["answer"],
            "correctness": correctness,
            "groundedness": grounded,
            "relevance": relevance,
            "retrieval_relevance": retr_rel,
        })
        print(f"[EVAL] {i}. corr={correctness} grounded={grounded} rel={relevance} retr_rel={retr_rel}")
    
    def _avg(key): return sum(1 for r in results if r[key]) / max(1, len(results))
    
    print("\n[EVAL][SUMMARY]")
    print("Correctness:", _avg("correctness"))
    print("Groundedness:", _avg("groundedness"))
    print("Relevance:", _avg("relevance"))
    print("Retrieval relevance:", _avg("retrieval_relevance"))
    return results
