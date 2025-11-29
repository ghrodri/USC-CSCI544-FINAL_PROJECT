import textwrap
from pathlib import Path
from typing import Dict, Tuple, List
import argparse

import torch
import torch.nn as nn
from transformers import (RobertaTokenizer,RobertaModel,AutoTokenizer,AutoModel,AutoModelForSeq2SeqLM)
from transformers.utils import logging as hf_logging

import logging

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

BASE_DIR = Path("./src/earning-calls/models")

SENTIMENT_MODEL_PATH = BASE_DIR/"sentiment.pt"
MULTITASK_CHECKPOINT_PATH = BASE_DIR / "multitask.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROBERTA_NAME = "roberta-base"
FINBERT_NAME = "ProsusAI/finbert"
SUMMARIZER_MODEL_NAME = "google/flan-t5-small"

SENTIMENT_ID2LABEL = {0: "negative",1: "neutral",2: "positive"}

UNCERTAINTY_ID2LABEL = {0: "none",1: "low",2: "medium",3: "high"}

FORWARD_ID2LABEL = {0: "none",1: "low",2: "medium",3: "high"}

class RoBERTaSentimentClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(ROBERTA_NAME)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_emb)
        return self.fc(x)

class FinBERTMultiTask(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.uncertainty_head = nn.Linear(hidden_size, num_labels)
        self.forward_head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits_unc = self.uncertainty_head(pooled)
        logits_fwd = self.forward_head(pooled)
        return logits_unc, logits_fwd


def load_all_models():
    sentiment_model = RoBERTaSentimentClassifier()
    sentiment_model.load_state_dict(torch.load(SENTIMENT_MODEL_PATH, map_location="cpu"))
    sentiment_model.to(DEVICE)
    sentiment_model.eval()
    sentiment_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_NAME)

    checkpoint = torch.load(MULTITASK_CHECKPOINT_PATH, map_location="cpu")
    label2id = checkpoint.get("label2id", {"none": 0, "low": 1, "medium": 2, "high": 3})
    num_labels = len(label2id)
    multitask_model = FinBERTMultiTask(FINBERT_NAME, num_labels)
    multitask_model.load_state_dict(checkpoint["model_state_dict"])
    multitask_model.to(DEVICE)
    multitask_model.eval()
    multitask_tokenizer = AutoTokenizer.from_pretrained(FINBERT_NAME)

    summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME).to(DEVICE)
    summarizer_model.eval()

    return (sentiment_model,sentiment_tokenizer,multitask_model,multitask_tokenizer,summarizer_tokenizer,summarizer_model)


def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    current = []
    current_len = 0
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def summarize_transcript(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    max_new_tokens: int = 128,
) -> str:
    chunks = chunk_text(text, max_chars=1200)
    partial_summaries = []
    for chunk in chunks:
        prompt = "summarize: " + chunk
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                length_penalty=1.0,
            )
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        partial_summaries.append(summary.strip())
    if len(partial_summaries) == 1:
        return partial_summaries[0]
    overall_prompt = "summarize: " + " ".join(partial_summaries)
    inputs = tokenizer(
        overall_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
        )
    final_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return final_summary.strip()


def predict_sentiment_chunk(text: str, model: RoBERTaSentimentClassifier, tokenizer: RobertaTokenizer, max_length: int = 256) -> Tuple[int, float]:
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


def predict_multitask_chunk(text: str, model: FinBERTMultiTask, tokenizer, max_length: int = 256) -> Tuple[int, float, int, float]:
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


def make_recommendation(sentiment_label: str, uncertainty_label: str, forward_label: str) -> str:
    if sentiment_label == "positive" and forward_label in {"medium", "high"} and uncertainty_label in {"none", "low"}:
        return "buy_more"
    if sentiment_label == "negative" and uncertainty_label in {"medium", "high"} and forward_label in {"none", "low"}:
        return "sell"
    return "hold"

def build_analysis_text(sentiment_label: str, uncertainty_label: str, forward_label: str) -> str:
    parts: List[str] = []
    if sentiment_label == "positive":
        parts.append("The overall tone of the call is positive.")
    elif sentiment_label == "negative":
        parts.append("The overall tone of the call is negative.")
    else:
        parts.append("The overall tone of the call is mostly neutral.")
    if uncertainty_label == "none":
        parts.append("Management expresses little uncertainty about the business.")
    elif uncertainty_label == "low":
        parts.append("There are some references to risk and uncertainty, but at a moderate level.")
    elif uncertainty_label == "medium":
        parts.append("There is a meaningful level of uncertainty in the guidance and risks discussed.")
    else:
        parts.append("The call reflects a high level of uncertainty and cautious language.")
    if forward_label == "none":
        parts.append("There is very little forward-looking communication or concrete guidance.")
    elif forward_label == "low":
        parts.append("The company provides only limited forward-looking commentary.")
    elif forward_label == "medium":
        parts.append("The company provides reasonably clear forward-looking guidance.")
    else:
        parts.append("The company strongly emphasizes future targets and projections.")
    return " ".join(parts)


def format_recommendation_text(rec_label: str) -> str:
    if rec_label == "buy_more":
        return "Recommendation: buy or increase the position, subject to the investor's risk profile."
    if rec_label == "sell":
        return "Recommendation: consider selling or reducing the position, given the tone and uncertainty."
    return "Recommendation: hold the position and keep monitoring new information."

def analyze_earnings_call_text(transcript: str,
    sentiment_model: RoBERTaSentimentClassifier,
    sentiment_tokenizer: RobertaTokenizer,
    multitask_model: FinBERTMultiTask,
    multitask_tokenizer,
    summarizer_tokenizer,
    summarizer_model) -> Dict[str, str]:
    chunks = chunk_text(transcript, max_chars=900)
    sent_ids, sent_confs = [], []
    unc_ids, unc_confs = [], []
    fwd_ids, fwd_confs = [], []
    for chunk in chunks:
        s_id, s_conf = predict_sentiment_chunk(chunk, sentiment_model, sentiment_tokenizer)
        u_id, u_conf, f_id, f_conf = predict_multitask_chunk(chunk, multitask_model, multitask_tokenizer)
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
    summary = summarize_transcript(transcript, summarizer_tokenizer, summarizer_model)
    analysis_text = build_analysis_text(sentiment_label, uncertainty_label, forward_label)
    rec_label = make_recommendation(sentiment_label, uncertainty_label, forward_label)
    rec_text = format_recommendation_text(rec_label)
    return {
        "summary": summary,
        "analysis": analysis_text,
        "recommendation": rec_text,
    }


def analyze_earnings_call_file(path: str) -> Dict[str, str]:
    (
        sentiment_model,
        sentiment_tokenizer,
        multitask_model,
        multitask_tokenizer,
        summarizer_tokenizer,
        summarizer_model,
    ) = load_all_models()
    transcript_path = Path(path)
    text = transcript_path.read_text(encoding="utf-8")
    return analyze_earnings_call_text(
        text,
        sentiment_model,
        sentiment_tokenizer,
        multitask_model,
        multitask_tokenizer,
        summarizer_tokenizer,
        summarizer_model,
    )


def main():
    parser = argparse.ArgumentParser(description="Earnings call analysis")
    parser.add_argument("--input-file", type=str, required=True)
    args = parser.parse_args()
    result = analyze_earnings_call_file(args.input_file)
    print("\n=== SUMMARY ===")
    print(textwrap.fill(result["summary"], width=100))
    print("\n=== ANALYSIS ===")
    print(textwrap.fill(result["analysis"], width=100))
    print("\n=== RECOMMENDATION ===")
    print(result["recommendation"])


if __name__ == "__main__":
    main()