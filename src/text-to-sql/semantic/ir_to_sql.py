import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# Resolve paths relative to this file so it works from CLI and when imported by backend.
_THIS_FILE = Path(__file__).resolve()
SEMANTIC_DIR = _THIS_FILE.parent
IR_SCHEMA_PATH = SEMANTIC_DIR / "ir_schema.json"
SEMANTIC_LAYER_PATH = SEMANTIC_DIR / "semantic_layer.yaml"


def load_ir_schema(path: Path = IR_SCHEMA_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_semantic_layer(path: Path = SEMANTIC_LAYER_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class SemanticError(Exception):
    pass

def get_metric_def(semantic: Dict[str, Any], name: str) -> Dict[str, Any]:
    m = semantic.get("metrics", {})
    if name not in m:
        raise SemanticError(f"Metric '{name}' not defined.")
    return m[name]

def get_intent_defaults(semantic: Dict[str, Any], intent: str) -> Dict[str, Any]:
    return semantic.get("intent_defaults", {}).get(intent, {})

def get_entity_def(semantic: Dict[str, Any], entity_name: str) -> Dict[str, Any]:
    entities = semantic.get("entities", {})
    if entity_name not in entities:
        raise SemanticError(f"Entity '{entity_name}' not found.")
    return entities[entity_name]

def resolve_default_entity_name(ir: Dict[str, Any], semantic: Dict[str, Any]) -> str:
    intent = ir.get("intent")
    metric_name = (ir.get("metric") or {}).get("name")

    intent_def = get_intent_defaults(semantic, intent)
    ent = intent_def.get("default_entity")

    if not ent and metric_name:
        ent = get_metric_def(semantic, metric_name).get("default_entity")

    if not ent:
        raise SemanticError(f"Cannot resolve entity for intent={intent}")

    return ent

def resolve_table_and_alias(semantic: Dict[str, Any], name: str):
    ent = get_entity_def(semantic, name)
    return ent["table"], name

def build_ticker_filter(ir: Dict[str, Any], alias: str) -> Optional[str]:
    entities = ir.get("entities") or {}
    assets: List[str] = entities.get("assets") or []
    if not assets:
        return None

    normalized = [a.lower().strip().replace(" ", "_") for a in assets]

    if "my_portfolio" in normalized or "my_holdings" in normalized:
        return f"{alias}.ticker IN (SELECT ticker FROM portfolio)"

    tickers_sql = ", ".join(f"'{t.upper()}'" for t in assets)
    return f"{alias}.ticker IN ({tickers_sql})"

def build_time_filter(ir: Dict[str, Any], alias: str) -> Optional[str]:
    time_spec = ir.get("time") or {}
    range_spec = time_spec.get("range") or {}
    t = range_spec.get("type")

    if not t or t == "all":
        return None

    if t == "year":
        year = range_spec.get("value")
        return f"EXTRACT(year FROM {alias}.date) = {year}"

    if t == "between_dates":
        start = range_spec.get("start")
        end = range_spec.get("end")
        parts = []
        if start:
            parts.append(f"{alias}.date >= DATE '{start}'")
        if end:
            parts.append(f"{alias}.date <= DATE '{end}'")
        return " AND ".join(parts)

    return None

def build_filters_from_ir(ir: Dict[str, Any], alias: str) -> List[str]:
    filters = ir.get("filters") or []
    clauses = []
    for f in filters:
        field = f.get("field")
        op = f.get("op", "=")
        val = f.get("value")

        col = f"{alias}.{field}"

        if isinstance(val, str):
            clauses.append(f"{col} {op} '{val}'")
        else:
            clauses.append(f"{col} {op} {val}")

    return clauses

def render_simple_metric(metric_def: Dict[str, Any], aliases: Dict[str, str]) -> str:
    expr = metric_def.get("expression")
    return expr.format(**aliases)

def render_template_metric(metric_def, ir, aliases):
    template = metric_def.get("sql_template")
    args = (ir.get("metric") or {}).get("arguments") or {}
    entities = ir.get("entities") or {}

    # Use your build_ticker_filter logic
    assets = entities.get("assets") or []
    normalized = [a.lower().strip().replace(" ", "_") for a in assets]

    if "my_portfolio" in normalized or "my_holdings" in normalized:
        tickers_sql = "(SELECT ticker FROM portfolio)"
    else:
        tickers_sql = ", ".join(f"'{t.upper()}'" for t in assets)

    year = args.get("year", 2024)
    month = args.get("month", 1)

    start = args.get("start_date", "2024-01-01")
    end = args.get("end_date", "2024-12-31")

    window = args.get("window", 30)

    params = {
        "tickers": tickers_sql,
        "year": year,
        "month": month,
        "start_date": f"DATE '{start}'",
        "end_date": f"DATE '{end}'",
        "window_size": window,
        **aliases,
    }

    return template.format(**params)

def render_metric_sql(ir: Dict[str, Any], semantic: Dict[str, Any], aliases: Dict[str, str]) -> str:
    metric = ir.get("metric") or {}
    name = metric.get("name")

    mdef = get_metric_def(semantic, name)

    # Template-based metric
    if "sql_template" in mdef:
        return render_template_metric(mdef, ir, aliases)

    # Expression-based metric
    if "expression" in mdef:
        expr = render_simple_metric(mdef, aliases)
        default_ent = mdef.get("default_entity")
        table, alias = resolve_table_and_alias(semantic, default_ent)

        where = []
        t1 = build_ticker_filter(ir, alias)
        if t1:
            where.append(t1)

        t2 = build_time_filter(ir, alias)
        if t2:
            where.append(t2)

        for w in build_filters_from_ir(ir, alias):
            where.append(w)

        wsql = "WHERE " + " AND ".join(where) if where else ""

        return f"""
        SELECT
          {alias}.ticker,
          {alias}.date,
          {expr} AS {name}
        FROM {table} AS {alias}
        {wsql}
        """

    raise SemanticError(f"Metric '{name}' has no implementation.")

def ir_to_sql(ir: Dict[str, Any], semantic: Optional[Dict[str, Any]] = None) -> str:
    if semantic is None:
        semantic = load_semantic_layer()

    ent_name = resolve_default_entity_name(ir, semantic)
    table, alias = resolve_table_and_alias(semantic, ent_name)

    aliases = {
        "prices_alias": alias,
        "returns_alias": alias,
        "div_yield_alias": alias,
        "dividends_alias": alias,
    }

    return render_metric_sql(ir, semantic, aliases).strip()

if __name__ == "__main__":
    ir = {
        "intent": "get_ytd_return",
        "entities": {"assets": ["AAPL"]},
        "metric": {"name": "ytd_return", "type": "return", "arguments": {"year": 2024}},
        "time": {"range": {"type": "year", "value": 2024}},
        "output": {"format": "table"}
    }

    sql = ir_to_sql(ir)
    print(sql)