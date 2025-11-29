# src/earning-calls/code/pipeline.py

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    AutoTokenizer,
    AutoModel,
)
from transformers.utils import logging as hf_logging

from dotenv import load_dotenv
from openai import OpenAI

hf_logging.set_verbosity_error()

load_dotenv()
client = OpenAI()

BASE_DIR = Path("./src/earning-calls/models")
SENTIMENT_MODEL_PATH = BASE_DIR / "sentiment.pt"
MULTITASK_CHECKPOINT_PATH = BASE_DIR / "multitask.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROBERTA_NAME = "roberta-base"
FINBERT_NAME = "ProsusAI/finbert"

SENTIMENT_ID2LABEL: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
UNCERTAINTY_ID2LABEL: Dict[int, str] = {0: "none", 1: "low", 2: "medium", 3: "high"}
FORWARD_ID2LABEL: Dict[int, str] = {0: "none", 1: "low", 2: "medium", 3: "high"}


class RoBERTaSentimentClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(ROBERTA_NAME)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids, attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.fc(self.dropout(cls))


class FinBERTMultiTask(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.uncertainty_head = nn.Linear(hidden, num_labels)
        self.forward_head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.uncertainty_head(pooled), self.forward_head(pooled)


def load_all_models():
    sentiment_model = RoBERTaSentimentClassifier()
    sentiment_model.load_state_dict(torch.load(SENTIMENT_MODEL_PATH, map_location="cpu"))
    sentiment_model.to(DEVICE).eval()
    sentiment_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_NAME)

    ckpt = torch.load(MULTITASK_CHECKPOINT_PATH, map_location="cpu")
    num_labels = 4
    multitask_model = FinBERTMultiTask(FINBERT_NAME, num_labels)
    multitask_model.load_state_dict(ckpt["model_state_dict"])
    multitask_model.to(DEVICE).eval()
    multitask_tokenizer = AutoTokenizer.from_pretrained(FINBERT_NAME)

    return sentiment_model, sentiment_tokenizer, multitask_model, multitask_tokenizer


def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    chunks: List[str] = []
    cur: List[str] = []
    length = 0
    for w in words:
        if length + len(w) + 1 > max_chars:
            chunks.append(" ".join(cur))
            cur = [w]
            length = len(w)
        else:
            cur.append(w)
            length += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def summarize_with_gpt(text: str) -> str:
    system_msg = (
        "You are a precise equity research analyst who writes short, informative earnings-call summaries "
        "for professional investors."
    )
    user_prompt = (
        "Summarize the following earnings call transcript in English in at most two paragraphs.\n"
        "- Focus on: headline numbers (revenue, EPS, growth rates), performance by products/segments/regions, "
        "key margin or cash-flow comments, guidance and overall tone.\n"
        "- Include key figures that are explicitly mentioned (with units and year-over-year growth when present).\n"
        "- Paraphrase; do not copy long sentences verbatim.\n\n"
        "Transcript:\n"
        f"{text}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()


def predict_sentiment_chunk(
    text: str,
    model: RoBERTaSentimentClassifier,
    tokenizer: RobertaTokenizer,
    max_length: int = 256,
) -> Tuple[int, float]:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1).cpu()
    label_id = int(torch.argmax(probs, dim=-1)[0])
    conf = float(torch.max(probs, dim=-1)[0])
    return label_id, conf


def predict_multitask_chunk(
    text: str,
    model: FinBERTMultiTask,
    tokenizer,
    max_length: int = 256,
) -> Tuple[int, float, int, float]:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits_unc, logits_fwd = model(enc["input_ids"], enc["attention_mask"])
        probs_unc = torch.softmax(logits_unc, dim=-1).cpu()
        probs_fwd = torch.softmax(logits_fwd, dim=-1).cpu()
    unc_id = int(torch.argmax(probs_unc, dim=-1)[0])
    unc_conf = float(torch.max(probs_unc, dim=-1)[0])
    fwd_id = int(torch.argmax(probs_fwd, dim=-1)[0])
    fwd_conf = float(torch.max(probs_fwd, dim=-1)[0])
    return unc_id, unc_conf, fwd_id, fwd_conf


def aggregate_predictions(ids: List[int], confidences: List[float]) -> int:
    if not ids:
        return 0
    scores: Dict[int, float] = {}
    for i, c in zip(ids, confidences):
        scores[i] = scores.get(i, 0.0) + c
    return max(scores.items(), key=lambda x: x[1])[0]


def build_analysis_text(sentiment_label: str, uncertainty_label: str, forward_label: str) -> str:
    parts: List[str] = []
    if sentiment_label == "positive":
        parts.append("The overall tone of the call is positive.")
    elif sentiment_label == "negative":
        parts.append("The overall tone of the call is negative.")
    else:
        parts.append("The overall tone of the call is mostly neutral.")
    if uncertainty_label == "none":
        parts.append("Management does not emphasize material uncertainty beyond normal business risks.")
    elif uncertainty_label == "low":
        parts.append("Management mentions some risks, but overall uncertainty appears contained.")
    elif uncertainty_label == "medium":
        parts.append("Management highlights a meaningful level of uncertainty around guidance and key risk factors.")
    else:
        parts.append("Management repeatedly stresses significant uncertainty and uses cautious language.")
    if forward_label == "none":
        parts.append("Forward-looking commentary is limited and mostly qualitative.")
    elif forward_label == "low":
        parts.append("Forward-looking commentary is present but high-level, with few precise targets.")
    elif forward_label == "medium":
        parts.append("The company provides reasonably clear forward-looking guidance and targets.")
    else:
        parts.append("The company places strong emphasis on detailed forward-looking targets and growth ambitions.")
    return " ".join(parts)


def make_recommendation(sentiment_label: str, uncertainty_label: str, forward_label: str) -> str:
    if sentiment_label == "positive" and forward_label in {"medium", "high"} and uncertainty_label in {"none", "low"}:
        return "Recommendation: the setup looks constructive; adding moderately to the position can be justified."
    if sentiment_label == "negative" and uncertainty_label in {"medium", "high"}:
        return "Recommendation: risk–reward looks weak; trimming or exiting the position is reasonable."
    return "Recommendation: maintain the current position and reassess after future quarters."


def analyze_earnings_call_text(
    text: str,
    sentiment_model: RoBERTaSentimentClassifier,
    sentiment_tokenizer: RobertaTokenizer,
    multitask_model: FinBERTMultiTask,
    multitask_tokenizer,
) -> Dict[str, str]:
    chunks = chunk_text(text, max_chars=900)
    sent_ids: List[int] = []
    sent_confs: List[float] = []
    unc_ids: List[int] = []
    unc_confs: List[float] = []
    fwd_ids: List[int] = []
    fwd_confs: List[float] = []
    for c in chunks:
        s_id, s_conf = predict_sentiment_chunk(c, sentiment_model, sentiment_tokenizer)
        u_id, u_conf, f_id, f_conf = predict_multitask_chunk(c, multitask_model, multitask_tokenizer)
        sent_ids.append(s_id)
        sent_confs.append(s_conf)
        unc_ids.append(u_id)
        unc_confs.append(u_conf)
        fwd_ids.append(f_id)
        fwd_confs.append(f_conf)
    agg_sent_id = aggregate_predictions(sent_ids, sent_confs)
    agg_unc_id = aggregate_predictions(unc_ids, unc_confs)
    agg_fwd_id = aggregate_predictions(fwd_ids, fwd_confs)
    sentiment_label = SENTIMENT_ID2LABEL[agg_sent_id]
    uncertainty_label = UNCERTAINTY_ID2LABEL[agg_unc_id]
    forward_label = FORWARD_ID2LABEL[agg_fwd_id]
    summary = summarize_with_gpt(text)
    analysis_text = build_analysis_text(sentiment_label, uncertainty_label, forward_label)
    recommendation_text = make_recommendation(sentiment_label, uncertainty_label, forward_label)
    return {
        "summary": summary,
        "analysis": analysis_text,
        "recommendation": recommendation_text,
    }

def run_pipeline(clean_transcript_text: str) -> Dict[str, str]:
    sentiment_model, sentiment_tokenizer, multitask_model, multitask_tokenizer = load_all_models()
    return analyze_earnings_call_text(
        clean_transcript_text,
        sentiment_model,
        sentiment_tokenizer,
        multitask_model,
        multitask_tokenizer,
    )