"""Campaign Assistant: a plain-language helper for the people who make the contacts.

Design rules (see the plan in the PR):
- Code does all the work. Sorting, filtering and counting happen in the tool functions below, so
  "top 10" is always exact. The language model only chooses a tool and phrases the result.
- Reasons are built from fixed templates over real customer fields. The model never invents a reason.
- Read-only: nothing here changes the list, the scenario or the models.
- Works without an API key: `answer_offline` routes the question with simple rules to the same tools.
- AI mode is switched on from the code base, not the UI, through environment variables or a `.env` file at
  the project root (git-ignored). Two providers: OpenAI (or any OpenAI-compatible endpoint, via
  OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL) and Anthropic (ANTHROPIC_API_KEY). LLM_PROVIDER picks one
  when both are set. No key = basic mode. Keys are never shown, logged or included in an error message.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from pacopt import config as cfg
from pacopt import optimize as op

ENV_FILE = cfg.PROJECT_ROOT / ".env"
ANTHROPIC_MODEL = "claude-haiku-5-5"
OPENAI_MODEL = "gpt-5.2"
MAX_ROWS = 50
MAX_TOOL_ROUNDS = 5
HISTORY_TURNS = 6

CHANNEL_LABEL = {"mobile_call": "Mobile call", "landline_call": "Landline call", "digital": "Digital"}
CHANNEL_ALIASES = {"mobile_call": ("mobile", "cell"), "landline_call": ("landline", "telephone"),
                   "digital": ("digital", "online", "email", "sms")}
CALL_CHANNELS = ("mobile_call", "landline_call")

GLOSSARY = {
    "expected profit": "The profit we expect from contacting this customer: the chance they subscribe, "
                       "multiplied by what a subscription is worth, minus the cost of the contact.",
    "propensity": "The model's estimate of how likely a customer is to say yes if contacted.",
    "segment": "A customer group: High value, Growing, Dormant or Low engagement.",
    "review": "Customers aged 75 or over are flagged. A manager should approve before you contact them.",
    "channel": "How the customer is contacted: mobile call, landline call or digital. It is fixed for each customer.",
    "contact cost": "What one contact costs us: £12 mobile call, £10 landline call, £2 digital by default.",
    "budget": "The most the whole campaign may spend on contacts.",
    "genetic algorithm": "The method that chose this list. It tries many possible lists and keeps improving the "
                         "best ones, while respecting every campaign limit.",
    "greedy": "A simple comparison method: rank customers by profit per pound of contact cost and keep adding them "
              "while the limits allow. It never goes back to reconsider an earlier choice.",
    "milp": "Mixed-Integer Linear Programming: an exact mathematical method that finds the provably best list "
            "under the same limits. We use it as a yardstick for the genetic algorithm.",
    "drift": "When today's customers start to look different from the customers the model learned from, "
             "so its predictions may become less reliable. The Monitoring page shows this with simulated shifts.",
    "evidently": "The tool behind the Monitoring page. It checks whether customer data and model scores have drifted.",
    "mlflow": "The tool behind the Experiments page. It records every model and optimiser run with its settings and results.",
    "rfm": "Recency, Frequency, Monetary: how recently, how often and how much a customer engaged. Here it is a proxy built "
           "from the last campaign contact, the number of previous contacts and the account balance.",
    "calibration": "Whether a predicted 20% chance really turns into about 20% yeses. Good calibration matters here "
                   "because the chances are multiplied by money.",
    "roc-auc": "How well the model ranks likely subscribers above unlikely ones. 0.5 is a coin flip, 1.0 is perfect.",
    "pr-auc": "Like ROC-AUC but focused on finding the real subscribers, who are a small share of customers.",
    "protected": "Customers aged 75 or over. They are not excluded, but they are capped at a small share of the list "
                 "and need a manager's approval before contact.",
    "eligible": "A customer we are allowed to contact: no credit default, no negative balance and not already contacted too often.",
    "uplift": "The extra subscriptions caused by contacting someone, compared with not contacting them. "
              "This project does not measure it, because there is no control group.",
}


# ----------------------------------------------------------------------------- the list the tools work on
@dataclass
class CampaignList:
    """The selected customers (best expected profit first) plus the figures the tools report."""
    strategy: str
    rows: pd.DataFrame
    table: pd.DataFrame                    # every customer in the scenario, to explain "why not"
    budget: float
    max_contacts: int
    comparison: op.Comparison | None = None
    cc: cfg.CampaignConstraints = cfg.CONSTRAINTS
    model_metrics: dict = field(default_factory=dict)
    drivers_fn: Callable[[], pd.DataFrame] | None = None     # lazy: permutation importance takes a few seconds

    @classmethod
    def build(cls, table: pd.DataFrame, comparison: op.Comparison, strategy: str, cc: cfg.CampaignConstraints,
              model_metrics: dict | None = None, drivers_fn: Callable[[], pd.DataFrame] | None = None) -> "CampaignList":
        pr = comparison.problem
        idx = pr.table_index[comparison.selections[strategy]]
        rows = table.loc[idx].sort_values("expected_profit", ascending=False)
        return cls(strategy, rows, table, cc.budget, cc.max_contacts, comparison, cc, model_metrics or {}, drivers_fn)


def reasons(row: pd.Series, limit: int = 3) -> list[str]:
    """Plain-language reasons, from fixed templates over real fields."""
    out: list[str] = []
    if row["prior_success"] == 1:
        out.append("said yes to a previous campaign")
    elif row["recency_bucket"] in ("<=90d", "91-180d"):
        out.append("was contacted recently, so the relationship is warm")
    if row["propensity"] >= 0.30:
        out.append(f"high chance of saying yes ({row['propensity']:.0%})")
    if row["balance_quintile"] >= 4:
        out.append("healthy account balance, so a larger deposit is likely")
    if row["ep_per_cost"] >= 5:
        out.append("low contact cost for the return expected")
    if not out:
        out.append(f"worth contacting: expected profit of £{row['expected_profit']:,.0f} after the contact cost")
    return out[:limit]


def _record(row: pd.Series) -> dict:
    return {"customer_id": row["customer_id"], "channel": CHANNEL_LABEL.get(row["channel"], row["channel"]),
            "segment": row["segment"], "chance_of_yes": round(float(row["propensity"]), 3),
            "expected_profit": round(float(row["expected_profit"]), 2), "contact_cost": round(float(row["contact_cost"]), 2),
            "needs_manager_review": bool(row["flag_age_75plus"]), "why": "; ".join(reasons(row))}


# ----------------------------------------------------------------------------- tools (read-only)
def _channels(channel: str | None) -> tuple[str, ...] | None:
    if not channel:
        return None
    c = channel.lower().strip().replace(" ", "_")
    if c in ("call", "calls", "phone", "phone_call"):
        return CALL_CHANNELS
    for key, aliases in CHANNEL_ALIASES.items():
        if c == key or any(a in c for a in aliases):
            return (key,)
    return ()


def top_customers(cl: CampaignList, n: int = 10, channel: str | None = None, segment: str | None = None) -> dict:
    rows, applied = cl.rows, {}
    chans = _channels(channel)
    if chans is not None:
        rows, applied["channel"] = rows[rows["channel"].isin(chans)], channel
    if segment:
        match = next((s for s in cfg.SEGMENTS if s.lower() == segment.lower().strip()), None)
        if match is None:
            return {"error": f"Unknown segment '{segment}'. Segments are: {', '.join(cfg.SEGMENTS)}."}
        rows, applied["segment"] = rows[rows["segment"] == match], match
    n = max(1, min(int(n), MAX_ROWS))
    top = rows.head(n)
    return {"filters": applied, "matching_customers_on_list": int(len(rows)), "returned": int(len(top)),
            "customers": [_record(r) for _, r in top.iterrows()]}


def customer(cl: CampaignList, customer_id: str) -> dict:
    cid = customer_id.strip().upper()
    if cid in set(cl.rows["customer_id"]):
        row = cl.rows[cl.rows["customer_id"] == cid].iloc[0]
        return {"on_list": True, "rank_on_list": int(cl.rows["customer_id"].tolist().index(cid)) + 1,
                "list_size": int(len(cl.rows)), **_record(row)}
    hit = cl.table[cl.table["customer_id"] == cid]
    if hit.empty:
        return {"error": f"No customer {cid} in this campaign universe."}
    row = hit.iloc[0]
    why_not = []
    if row["eligible"] != 1:
        if row["elig_default"]:
            why_not.append("has a credit default on record")
        if row["elig_negative_balance"]:
            why_not.append("has a negative balance")
        if row["elig_fatigue"]:
            why_not.append("has already been contacted too many times")
    elif row["expected_profit"] <= 0:
        why_not.append("the expected profit is below the contact cost")
    else:
        why_not.append("eligible, but others offered a better return within the budget and limits")
    return {"on_list": False, "customer_id": cid, "why_not_on_list": why_not}


def flagged(cl: CampaignList, n: int = 10) -> dict:
    f = cl.rows[cl.rows["flag_age_75plus"] == 1]
    return {"count_needing_review": int(len(f)), "returned": int(min(len(f), n)),
            "customers": [_record(r) for _, r in f.head(max(1, min(int(n), MAX_ROWS))).iterrows()]}


def list_summary(cl: CampaignList) -> dict:
    r = cl.rows
    return {"strategy": cl.strategy, "customers_on_list": int(len(r)), "max_contacts": cl.max_contacts,
            "total_contact_cost": round(float(r["contact_cost"].sum()), 2), "budget": cl.budget,
            "total_expected_profit": round(float(r["expected_profit"].sum()), 2),
            "by_channel": {CHANNEL_LABEL.get(k, k): int(v) for k, v in r["channel"].value_counts().items()},
            "by_segment": {k: int(v) for k, v in r["segment"].value_counts().items()},
            "needing_manager_review": int(r["flag_age_75plus"].sum())}


def constraints(cl: CampaignList) -> dict:
    """Every campaign limit with the list's value against it, and whether it is tight (within 2% of the limit)."""
    chk = op.check_constraints(cl.table, cl.rows["customer_id"], cl.cc)
    out = []
    for _, r in chk.iterrows():
        name = r["constraint"]
        if name in ("Unique customers", "Known customers", "Eligible customers only"):
            continue
        is_min = name.startswith("Min share")
        tight = bool(r["value"] <= r["limit"] * 1.02) if is_min else bool(r["value"] >= r["limit"] * 0.98)
        pct = name.startswith(("Min share", "Max share"))
        out.append({"limit_name": name, "list_value": round(float(r["value"]), 4 if pct else 2),
                    "limit": round(float(r["limit"]), 4 if pct else 2), "met": bool(r["ok"]), "tight": tight and bool(r["ok"])})
    return {"all_limits_met": bool(all(o["met"] for o in out)), "limits": out,
            "tight_limits": [o["limit_name"] for o in out if o["tight"]]}


def compare_strategies(cl: CampaignList) -> dict:
    """How the strategies compare on expected profit, so staff can see why this list was chosen."""
    if cl.comparison is None:
        return {"error": "No comparison available."}
    t = cl.comparison.table
    rows = []
    for name, r in t.iterrows():
        rows.append({"strategy": name, "customers": int(r["contacts"]), "spend": round(float(r["spend"]), 0),
                     "expected_profit": round(float(r["expected_contribution"]), 0), "limits_met": bool(r["feasible"]),
                     **({"share_of_best_possible": round(float(r["pct_of_optimum"]), 3)} if "pct_of_optimum" in t and
                        r["pct_of_optimum"] == r["pct_of_optimum"] else {})})
    return {"list_in_use": cl.strategy, "strategies": rows,
            "notes": "Random = untargeted. Propensity = rank by likelihood only. Greedy = rank by profit per pound. "
                     "Genetic Algorithm = the optimised list. MILP = exact best possible, used as a yardstick."}


def model_info(cl: CampaignList) -> dict:
    """How good the prediction model is and what drives its predictions."""
    mm = cl.model_metrics
    if not mm:
        return {"error": "Model details are not available."}
    sel = mm.get(mm.get("selected", "lr"), {})
    out = {"model": sel.get("name"), "why_this_model": mm.get("reason"),
           "roc_auc": round(sel.get("roc_auc", float("nan")), 3), "pr_auc": round(sel.get("pr_auc", float("nan")), 3),
           "overall_subscribe_rate": round(sel.get("base_rate", float("nan")), 3),
           "max_calibration_gap_pct_points": round(sel.get("max_calibration_gap_pp", float("nan")), 2),
           "hit_rate_in_top_500": (sel.get("precision_at_k") or {}).get("K=500")}
    if cl.drivers_fn is not None:
        try:
            d = cl.drivers_fn().head(6)
            out["top_drivers"] = [x.replace("_", " ") for x in d["feature"]]
        except Exception:
            pass
    return out


def explain_term(term: str) -> dict:
    t = term.lower().strip()
    for k, v in GLOSSARY.items():
        if k in t or t in k:
            return {"term": k, "meaning": v}
    return {"error": f"No definition for '{term}'.", "known_terms": sorted(GLOSSARY)}


TOOL_FUNCS = {"top_customers": top_customers, "customer": customer, "flagged": flagged,
              "list_summary": list_summary, "constraints": constraints, "compare_strategies": compare_strategies,
              "model_info": model_info, "explain_term": explain_term}
LIST_TOOLS = {"top_customers", "flagged"}      # results shown as a table in the UI


def run_tool(cl: CampaignList, name: str, args: dict) -> dict:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name}."}
    try:
        return fn(**args) if name == "explain_term" else fn(cl, **args)
    except (TypeError, ValueError) as e:
        return {"error": f"Bad request: {e}"}


TOOLS = [
    {"name": "top_customers", "description": "The best customers on the campaign list to contact, highest expected profit first. "
     "Optionally filter by channel (call, mobile, landline, digital) or segment.",
     "input_schema": {"type": "object", "properties": {
         "n": {"type": "integer", "description": "How many customers (default 10, max 50)"},
         "channel": {"type": "string", "description": "call, mobile, landline or digital"},
         "segment": {"type": "string", "description": "High value, Growing, Dormant or Low engagement"}}}},
    {"name": "customer", "description": "Look up one customer by ID (like C100245): is the customer on the list and why or why not.",
     "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}},
    {"name": "flagged", "description": "Customers on the list aged 75 or over, who need a manager's approval before contact.",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    {"name": "list_summary", "description": "Size, total cost, budget, expected profit and channel and segment split of the list.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "constraints", "description": "Every campaign limit (budget, contacts, channel capacity, segment shares, 75+ cap), "
     "the list's value against each, and which limits are tight.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "compare_strategies", "description": "Expected profit, size and spend of each strategy (Random, Propensity, Greedy, "
     "Genetic Algorithm, MILP): why this list was chosen over simpler ones.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "model_info", "description": "How accurate the prediction model is, why it was chosen, and its top drivers.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "explain_term", "description": "Plain-language meaning of a term (expected profit, segment, greedy, MILP, drift, "
     "calibration, RFM, uplift, protected customers and more).",
     "input_schema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]}},
]

SYSTEM = """You are the Campaign Assistant inside a bank's term-deposit campaign tool. Your users are staff who make the
customer contacts, and sometimes their managers. Speak plainly, as to a colleague who is not technical, and keep answers short.

What the tool does (for general questions): it predicts each customer's chance of opening a term deposit, works out the
expected profit of contacting them (chance x value of a deposit - contact cost), then chooses which customers to contact so
the whole campaign earns the most profit while respecting every limit: budget, maximum contacts, call and digital capacity,
minimum and maximum shares for each customer group, and a small cap on customers aged 75 and over. The channel for each
customer is fixed by how we can reach them. The figures are illustrative assumptions from a public 2008-2010 Portuguese
dataset, not real bank results.

Rules:
- Use ONLY tool results for any customer, number or fact about this campaign. If the tools do not give it, say you do not have it.
- Never invent reasons. Quote the reasons the tools return.
- Avoid jargon by default. If the user asks about a technical term or how the model or optimiser works, explain it simply,
  using explain_term or model_info.
- Customers with needs_manager_review = true are aged 75 or over. Always say a manager must approve before contacting them.
- When you list customers, give the IDs and a one-line reason each. The app also shows the table.
- You can only read the campaign data. You cannot change the list, contact anyone, or run anything.
- Stay on topic: this campaign, its customers, how the list was chosen, the limits, the model and the project's concepts.
  For anything else (general knowledge, other products, personal or financial advice) politely say you can only help
  with the campaign."""


# ----------------------------------------------------------------------------- answers
@dataclass
class Answer:
    text: str
    rows: list[dict] = field(default_factory=list)
    mode: str = "offline"


def _why_failed(e: Exception) -> str:
    """A short, key-free reason. Never includes the exception text, which could echo request details."""
    name = type(e).__name__
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return "the API key was not accepted"
    if name == "RateLimitError":
        return "the rate limit was reached"
    if name in ("APIConnectionError", "APITimeoutError"):
        return "no connection to the AI service"
    if name == "NotFoundError":
        return "the endpoint or model/deployment name was not found (check OPENAI_BASE_URL and OPENAI_MODEL)"
    return f"unexpected error ({name})"


def setting(name: str) -> str | None:
    """A setting from the environment, else from `.env` at the project root. Read on every question, so a change
    takes effect on the next question (no restart needed). Empty values count as unset."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        for line in ENV_FILE.read_text().splitlines():
            key, sep, val = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == name:
                val = val.strip()
                return None if val.startswith("#") else (val.split(" #")[0].strip().strip("'\"") or None)
    except OSError:
        pass
    return None


@dataclass(frozen=True)
class Provider:
    name: str               # "openai", "azure" or "anthropic"
    api_key: str
    model: str              # for Azure: the deployment name
    base_url: str | None = None
    api_version: str | None = None    # Azure only

    def __repr__(self) -> str:        # never print the key
        return (f"Provider(name={self.name!r}, model={self.model!r}, base_url={self.base_url!r}, "
                f"api_version={self.api_version!r})")


AZURE_API_VERSION = "2025-04-01-preview"


def _openai_provider(key: str) -> Provider:
    """OpenAI, an OpenAI-compatible gateway, or Azure OpenAI. For Azure any copied URL works (for example
    .../openai/responses?api-version=...): only the host and the api-version are used."""
    from urllib.parse import parse_qs, urlparse
    url, model = setting("OPENAI_BASE_URL"), setting("OPENAI_MODEL") or OPENAI_MODEL
    if url and urlparse(url).hostname and urlparse(url).hostname.endswith((".azure.com", ".azure.us")):
        u = urlparse(url)
        version = (parse_qs(u.query).get("api-version") or [None])[0] or setting("OPENAI_API_VERSION") or AZURE_API_VERSION
        return Provider("azure", key, model, f"{u.scheme}://{u.netloc}", version)
    return Provider("openai", key, model, url)


def provider() -> Provider | None:
    """The configured LLM, or None for basic mode. LLM_PROVIDER wins; otherwise the first key found (OpenAI first)."""
    choice = (setting("LLM_PROVIDER") or "").lower()
    oa, an = setting("OPENAI_API_KEY"), setting("ANTHROPIC_API_KEY")
    if choice in ("none", "off", "basic"):
        return None
    if oa and choice in ("", "openai", "azure"):
        return _openai_provider(oa)
    if an and choice in ("", "anthropic"):
        return Provider("anthropic", an, setting("ANTHROPIC_MODEL") or ANTHROPIC_MODEL, setting("ANTHROPIC_BASE_URL"))
    return None


def ai_enabled() -> bool:
    return provider() is not None


def ask(cl: CampaignList, question: str, history: list[dict] | None = None) -> Answer:
    """AI answer when a provider is configured, otherwise (or on any API failure) the offline router."""
    p = provider()
    if p is None:
        return answer_offline(cl, question)
    try:
        return (_ask_anthropic if p.name == "anthropic" else _ask_openai)(cl, question, history or [], p)
    except Exception as e:       # network, auth, quota: never break the page
        fallback = answer_offline(cl, question)
        fallback.text += f"\n\n_(AI assistant unavailable: {_why_failed(e)}. Showing a basic answer instead.)_"
        return fallback


def _ask_anthropic(cl: CampaignList, question: str, history: list[dict], p: Provider) -> Answer:
    import anthropic
    client = anthropic.Anthropic(api_key=p.api_key, base_url=p.base_url)
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-HISTORY_TURNS:]]
    messages.append({"role": "user", "content": question})
    rows: list[dict] = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(model=p.model, max_tokens=700, system=SYSTEM, tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return Answer(text or "Sorry, I could not answer that.", rows, "ai")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                out = run_tool(cl, b.name, b.input or {})
                if b.name in LIST_TOOLS and out.get("customers"):
                    rows = out["customers"]
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    return Answer("Sorry, that took too many steps. Try a simpler question.", rows, "ai")


OPENAI_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in TOOLS]


def _ask_openai(cl: CampaignList, question: str, history: list[dict], p: Provider) -> Answer:
    """Chat Completions with function calling: supported by OpenAI and most OpenAI-compatible endpoints."""
    import openai
    if p.name == "azure":
        client = openai.AzureOpenAI(api_key=p.api_key, azure_endpoint=p.base_url, api_version=p.api_version)
    else:
        client = openai.OpenAI(api_key=p.api_key, base_url=p.base_url)
    messages = [{"role": "system", "content": SYSTEM}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-HISTORY_TURNS:]]
    messages.append({"role": "user", "content": question})
    rows: list[dict] = []
    for _ in range(MAX_TOOL_ROUNDS):
        # max_completion_tokens (not max_tokens) for GPT-5 models; generous because reasoning uses some of it
        resp = client.chat.completions.create(model=p.model, messages=messages, tools=OPENAI_TOOLS,
                                              max_completion_tokens=2_000)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return Answer((msg.content or "").strip() or "Sorry, I could not answer that.", rows, "ai")
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out = run_tool(cl, tc.function.name, args if isinstance(args, dict) else {})
            if tc.function.name in LIST_TOOLS and out.get("customers"):
                rows = out["customers"]
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
    return Answer("Sorry, that took too many steps. Try a simpler question.", rows, "ai")


# ----------------------------------------------------------------------------- offline router
def _line(c: dict) -> str:
    review = " **(manager approval needed)**" if c["needs_manager_review"] else ""
    return f"- **{c['customer_id']}** ({c['channel']}, {c['segment']}): {c['why']}{review}"


def answer_offline(cl: CampaignList, question: str) -> Answer:
    q = question.lower()
    cid = re.search(r"c\d{6}", q)
    if cid:
        r = customer(cl, cid.group())
        if "error" in r:
            return Answer(r["error"])
        if r["on_list"]:
            note = " A manager must approve before you contact them." if r["needs_manager_review"] else ""
            return Answer(f"**{r['customer_id']}** is number {r['rank_on_list']} of {r['list_size']} on the list "
                          f"({r['channel']}, {r['segment']}). Why: {r['why']}.{note}")
        return Answer(f"**{r['customer_id']}** is not on the list: " + "; ".join(r["why_not_on_list"]) + ".")
    if re.search(r"review|approv|75\s*\+|aged 75|over 75|elderly|flag|manager", q):
        r = flagged(cl)
        text = (f"{r['count_needing_review']} customers on the list are aged 75 or over and need a manager's "
                "approval before you contact them.")
        return Answer(text + ("\n\n" + "\n".join(_line(c) for c in r["customers"]) if r["customers"] else ""), r["customers"])
    if re.search(r"what is|what does|what's|what are|meaning|mean\b|explain|define", q):
        r = explain_term(q)
        if "meaning" in r:
            return Answer(f"**{r['term'].title()}:** {r['meaning']}")
    if re.search(r"constraint|limit|binding|tight|rule", q):
        r = constraints(cl)
        def fmt(name, v):
            return f"{v:.0%}" if "share" in name.lower() else (f"£{v:,.0f}" if "£" in name else f"{v:,.0f}")
        lines = [f"- {o['limit_name'].replace(' (£)', '')}: {fmt(o['limit_name'], o['list_value'])} vs limit "
                 f"{fmt(o['limit_name'], o['limit'])}" + (" (tight)" if o["tight"] else "") for o in r["limits"]]
        head = ("Every campaign limit is met." if r["all_limits_met"] else "Some limits are NOT met.") + \
               (f" The tightest are: {', '.join(r['tight_limits'])}." if r["tight_limits"] else "")
        return Answer(head + "\n\n" + "\n".join(lines))
    if re.search(r"compare|greedy|genetic|strateg|baseline|better|why .*(chosen|choose|this list)|random", q):
        r = compare_strategies(cl)
        lines = [f"- **{x['strategy']}**: expected profit £{x['expected_profit']:,.0f}, {x['customers']:,} customers"
                 for x in r["strategies"]]
        return Answer(f"Strategies compared on expected profit (list in use: {r['list_in_use']}):\n\n" + "\n".join(lines) +
                      f"\n\n{r['notes']}")
    if re.search(r"model|accura|driver|factor|predict|propensity score", q):
        r = model_info(cl)
        if "error" in r:
            return Answer(r["error"])
        text = (f"The prediction model is **{r['model']}**. Its ROC-AUC is {r['roc_auc']} and PR-AUC {r['pr_auc']} (the overall "
                f"subscribe rate is {r['overall_subscribe_rate']:.0%}); predicted chances are within "
                f"{r['max_calibration_gap_pct_points']} points of reality. {r['why_this_model']}.")
        if r.get("top_drivers"):
            text += " Biggest drivers: " + ", ".join(r["top_drivers"]) + "."
        return Answer(text)
    if re.search(r"how many|total|cost|spend|budget|summary|size|overview", q) and not re.search(r"top|best", q):
        s = list_summary(cl)
        return Answer(f"The list has **{s['customers_on_list']:,}** customers (limit {s['max_contacts']:,}). Total contact cost is "
                      f"**£{s['total_contact_cost']:,.0f}** of a £{s['budget']:,.0f} budget, with expected profit of "
                      f"**£{s['total_expected_profit']:,.0f}**. Channels: "
                      + ", ".join(f"{k} {v:,}" for k, v in s["by_channel"].items()) + ". "
                      f"{s['needing_manager_review']} customers need manager review.")
    n = re.search(r"\b(\d{1,2})\b", q)
    channel = next((w for w in ("mobile", "landline", "digital", "online", "call", "phone") if w in q), None)
    segment = next((s for s in cfg.SEGMENTS if s.lower() in q), None)
    if re.search(r"top|best|first|who|contact|call|list|priorit|start", q):
        r = top_customers(cl, int(n.group(1)) if n else 10, channel, segment)
        if "error" in r:
            return Answer(r["error"])
        if not r["customers"]:
            return Answer("No customers on the list match that.")
        head = f"Here are the top {r['returned']}" + (f" ({', '.join(r['filters'].values())})" if r["filters"] else "") + \
               ", best expected profit first:"
        return Answer(head + "\n\n" + "\n".join(_line(c) for c in r["customers"]), r["customers"])
    return Answer("I can help with your contact list: try *top 10 customers to call*, *who needs manager approval*, "
                  "*why is C100245 on my list*, *which limits are tight*, *why this list over Greedy*, *how accurate is the model*, "
                  "or *what is expected profit*. (Connect an AI key on this page for open-ended questions.)")
