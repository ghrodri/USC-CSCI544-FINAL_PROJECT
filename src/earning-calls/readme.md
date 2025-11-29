# Earnings Call Analysis System

This project ingests **earnings-call PDFs**, cleans and extracts the transcript, summarizes the content using an LLM, and classifies the call using two custom financial NLP models:

1. **Sentiment Model (RoBERTa)**  
   Determines whether the tone of the call is *positive*, *neutral*, or *negative*.

2. **Uncertainty & Forward-Looking Model (FinBERT Multi-Task)**  
   Detects the degree of *uncertainty* and *forward-looking guidance* expressed by management.

3. **Final Summary + Analysis + Recommendation**  
   Produces a 1–2 paragraph executive summary, interprets the model outputs, and generates a natural buy/hold/sell-style recommendation.

---
## Run the Pipeline
python src/earning-calls/code/run_all.py \
  --pdf "path to earnings-call.pdf" \
  --out "path to output.json"

---
## Output Format

The system returns a structured JSON object:

```json
{
  "summary": "Apple delivered strong Q4 results... (2 paragraphs)",
  "analysis": "The overall tone of the call is positive...",
  "recommendation": "Recommendation: hold the position."
}