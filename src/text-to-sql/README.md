# Financial Text-to-SQL System Using Qwen + DuckDB
## Overview
The system accepts free-form financial questions such as:
```
What is the year-to-date return of my portfolio for 2024?
```
and produces:
A structured **Intermediate Representation (IR)**
A deterministic **SQL query** generated from the IR
The **execution result** from DuckDB
A final **Plain Language answer**
The project uses:
Qwen 2.5-0.5B fine-tuned with LoRA adapters 
A controlled JSON IR schema 
A semantic IR to SQL compiler 
A DuckDB financial warehouse 
Optional ingestion of user portfolio data 
---
## System Architecture
| Component | Description |
|----------|-------------|
| Qwen 2.5-0.5B + LoRA | IR generation from user questions |
| IR Schema | JSON-based structure enforced during training |
| Semantic Layer | Deterministic IR→SQL rules |
| DuckDB | Local financial database |
| Natural Answer | Converts dataframe output into readable text |
---
## Setup Instructions
### 1. Build the Local DuckDB Schema (Required)
Before anything else, run:
```
python src/etl/build_schema.py
```
This step configures all table paths so the warehouse works correctly on the local machine.
---
### 2. Load User Portfolio Data (Optional) 
If the user wants to ask:
```
What is the year-to-date return of my portfolio?
```
They must upload a CSV with the following columns:
```
ticker,shares,avg_cost
AAPL,10,150
AMZN,5,120
...
```
Then execute:
```
python src/temp/temp_table.py path/to/user_portfolio.csv
```
This creates the internal DuckDB table:
```
portfolio(ticker, shares, avg_cost)
```
---
## Running Inference
### Using the base pipeline:
```
python src/text-to-sql/code/run_qwen_pipeline.py "What is the year-to-date return of NTRS for 2024?"
```
---
## Evaluation Summary
We evaluated the system on 1,000 held-out examples.
```
IR Exact Match:         1.000
Intent Accuracy:        1.000
Metric Accuracy:        1.000
Time Accuracy:          1.000
Assets Precision:       1.000
Assets Recall:          1.000
Assets F1:              1.000
SQL Execution Success:  1.000
SQL Result Match:       1.000
```
These results confirm the IR model and semantic layer are perfectly aligned.
---
## Example Output
Input:
```
What is the year-to-date return of my portfolio for 2024?
```
Generated SQL:
```
WITH prices AS (
  SELECT
      ticker,
      date,
      close,
      FIRST_VALUE(close) OVER (
          PARTITION BY ticker
          ORDER BY date
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS first_close_ytd
  FROM core.prices_daily
  WHERE ticker IN (SELECT ticker FROM portfolio)
    AND EXTRACT(year FROM date) = 2024
)
SELECT
    ticker,
    MAX(date) AS as_of_date,
    MAX(close / first_close_ytd - 1) AS ytd_return
FROM prices
GROUP BY ticker;
```
Example natural-language answer:
```
The year-to-date return for Apple Inc. is 15.32%.
```
---
## Summary
After the database schema is built and (optionally) user portfolios are loaded, the entire system supports plain language queries (Only English), IR generation, SQL translation, and final answer generation—all via a single function call.