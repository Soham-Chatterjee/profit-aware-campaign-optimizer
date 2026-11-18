"""Campaign Assistant tests: tools are exact and read-only, reasons come from templates, the offline router
works without a key, and a failing AI call falls back instead of raising."""
import pytest
from streamlit.testing.v1 import AppTest

from pacopt import assistant as a
from pacopt import optimize as op
from pacopt import scenario as sc
from pathlib import Path


@pytest.fixture(scope="module")
def cl():
    t = sc.build_table(sc.DEFAULT)
    comp = op.compare_strategies(t, sc.DEFAULT.constraints, run_genetic=False, run_milp=False)
    return a.CampaignList.build(t, comp, "Greedy", sc.DEFAULT.constraints)


def test_top_customers_exact_and_sorted(cl):
    r = a.top_customers(cl, 10)
    ep = [c["expected_profit"] for c in r["customers"]]
    assert r["returned"] == 10 and ep == sorted(ep, reverse=True)
    assert r["customers"][0]["customer_id"] == cl.rows.iloc[0]["customer_id"]


def test_filters_and_limits(cl):
    calls = a.top_customers(cl, 5, channel="phone")
    assert {c["channel"] for c in calls["customers"]} <= {"Mobile call", "Landline call"}
    seg = a.top_customers(cl, 5, segment="dormant")
    assert {c["segment"] for c in seg["customers"]} == {"Dormant"}
    assert a.top_customers(cl, 10_000)["returned"] <= a.MAX_ROWS
    assert "error" in a.top_customers(cl, 5, segment="nonsense")


def test_every_customer_has_a_reason_and_no_private_fields(cl):
    for c in a.top_customers(cl, 20)["customers"]:
        assert c["why"]
        assert set(c) == {"customer_id", "channel", "segment", "chance_of_yes", "expected_profit", "contact_cost",
                          "needs_manager_review", "why"}


def test_customer_lookup_on_and_off_list(cl):
    on = a.customer(cl, cl.rows.iloc[0]["customer_id"].lower())
    assert on["on_list"] and on["rank_on_list"] == 1
    off_id = cl.table[~cl.table["customer_id"].isin(cl.rows["customer_id"])].iloc[0]["customer_id"]
    off = a.customer(cl, off_id)
    assert not off["on_list"] and off["why_not_on_list"]
    assert "error" in a.customer(cl, "C999999")


def test_flagged_and_summary_match_the_list(cl):
    assert a.flagged(cl)["count_needing_review"] == int(cl.rows["flag_age_75plus"].sum())
    s = a.list_summary(cl)
    assert s["customers_on_list"] == len(cl.rows) and s["total_contact_cost"] <= cl.budget + 1e-6


def test_run_tool_rejects_unknown_and_bad_args(cl):
    assert "error" in a.run_tool(cl, "delete_everything", {})
    assert "error" in a.run_tool(cl, "top_customers", {"bogus": 1})


@pytest.mark.parametrize("q,needle", [("Give me the top 10 customers I should contact", "top 10"),
                                      ("top 3 for mobile calls", "top 3"),
                                      ("who needs manager approval?", "aged 75"),
                                      ("how many customers and what will it cost", "budget"),
                                      ("what is expected profit", "Expected Profit"),
                                      ("tell me a joke", "I can help with your contact list")])
def test_offline_router(cl, q, needle):
    assert needle.lower() in a.answer_offline(cl, q).text.lower()


def test_offline_lists_return_table_rows(cl, no_key):
    ans = a.answer_offline(cl, "top 10 customers")
    assert len(ans.rows) == 10 and ans.mode == "offline"
    cid = cl.rows.iloc[2]["customer_id"]
    assert cid in a.answer_offline(cl, f"why is {cid} on my list?").text


@pytest.fixture
def no_key(monkeypatch, tmp_path):
    for v in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "ANTHROPIC_MODEL",
              "ANTHROPIC_BASE_URL", "LLM_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(a, "ENV_FILE", tmp_path / ".env")
    return tmp_path / ".env"


def test_no_key_uses_offline_and_ai_failure_falls_back_without_leaking_the_key(cl, monkeypatch, no_key):
    assert a.ask(cl, "top 5").mode == "offline"
    secret = "sk-ant-SECRET-123"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    def boom(*args, **kw):
        raise RuntimeError(f"request failed for key {secret}")
    monkeypatch.setattr(a, "_ask_anthropic", boom)
    ans = a.ask(cl, "top 5")
    assert ans.mode == "offline" and "unavailable" in ans.text and ans.rows
    assert secret not in ans.text and "request failed" not in ans.text


def test_provider_from_env_var_or_dotenv_file(monkeypatch, no_key):
    assert a.provider() is None and not a.ai_enabled()
    no_key.write_text('# comment\nOTHER=1\nexport OPENAI_API_KEY="sk-oa-file"\n'
                      'OPENAI_BASE_URL=          # leave empty for api.openai.com\nOPENAI_MODEL=gpt-5.2   # model\n')
    p = a.provider()
    assert (p.name, p.api_key, p.model, p.base_url) == ("openai", "sk-oa-file", "gpt-5.2", None)
    assert "sk-oa-file" not in repr(p)                          # keys never printed
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa-env")
    assert a.provider().api_key == "sk-oa-env"                  # the environment wins over the file
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    assert a.provider().base_url == "https://gateway.example/v1"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    assert a.provider().name == "openai"                        # OpenAI first when both are set...
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert a.provider().name == "anthropic"                     # ...unless LLM_PROVIDER says otherwise
    monkeypatch.setenv("LLM_PROVIDER", "basic")
    assert a.provider() is None


def test_new_tools(cl):
    c = a.constraints(cl)
    assert c["all_limits_met"] and any(o["limit_name"].startswith("Budget") for o in c["limits"])
    assert all("limit_name" in o and "met" in o for o in c["limits"])
    cmp_ = a.compare_strategies(cl)
    assert {x["strategy"] for x in cmp_["strategies"]} >= {"Random", "Propensity", "Greedy"}
    assert a.model_info(cl).get("error")                      # no metrics supplied in this fixture
    cl2 = a.CampaignList(**{**cl.__dict__, "model_metrics": {"selected": "lr", "reason": "r", "lr": {
        "name": "Logistic Regression", "roc_auc": 0.75, "pr_auc": 0.38, "base_rate": 0.117,
        "max_calibration_gap_pp": 1.2, "precision_at_k": {"K=500": 0.57}}},
        "drivers_fn": lambda: __import__("pandas").DataFrame({"feature": ["poutcome", "contact_type"]})})
    mi = a.model_info(cl2)
    assert mi["model"] == "Logistic Regression" and mi["top_drivers"] == ["poutcome", "contact type"]
    for term in ("greedy", "milp", "drift", "uplift", "rfm"):
        assert "meaning" in a.explain_term(term)


@pytest.mark.parametrize("q,needle", [("which limits are tight?", "limit"), ("why is this better than greedy", "Strategies compared"),
                                      ("what is MILP", "Mixed-Integer"), ("what is greedy", "profit per pound")])
def test_offline_router_new_questions(cl, q, needle):
    assert needle.lower() in a.answer_offline(cl, q).text.lower()


def test_anthropic_tool_loop_with_fake_client(cl, monkeypatch, no_key):
    """Drives the real tool loop with a scripted client: one tool call, then a text answer."""
    from types import SimpleNamespace as NS
    import anthropic
    calls, seen = [], {}

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw["messages"][-1])
            if len(calls) == 1:
                block = NS(type="tool_use", id="t1", name="top_customers", input={"n": 3})
                return NS(stop_reason="tool_use", content=[block])
            return NS(stop_reason="end_turn", content=[NS(type="text", text="Here are your 3.")])

    def fake_client(api_key=None, base_url=None):
        seen["key"] = api_key
        return NS(messages=FakeMessages())
    monkeypatch.setattr(anthropic, "Anthropic", fake_client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-typed")
    ans = a.ask(cl, "top 3 please")
    assert ans.mode == "ai" and ans.text == "Here are your 3." and len(ans.rows) == 3
    assert seen["key"] == "sk-ant-typed" and calls[1]["content"][0]["type"] == "tool_result"


def test_openai_tool_loop_with_fake_client(cl, monkeypatch, no_key):
    """Drives the OpenAI function-calling loop with a scripted client: one tool call, then a text answer."""
    from types import SimpleNamespace as NS
    import openai
    calls, seen = [], {}

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                tc = NS(id="c1", function=NS(name="top_customers", arguments='{"n": 4, "channel": "phone"}'))
                return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))])
            return NS(choices=[NS(message=NS(content="Your 4 calls.", tool_calls=None))])

    def fake_client(api_key=None, base_url=None):
        seen.update(key=api_key, url=base_url)
        return NS(chat=NS(completions=FakeCompletions()))
    monkeypatch.setattr(openai, "OpenAI", fake_client)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    ans = a.ask(cl, "top 4 phone calls")
    assert ans.mode == "ai" and ans.text == "Your 4 calls." and len(ans.rows) == 4
    assert {r["channel"] for r in ans.rows} <= {"Mobile call", "Landline call"}
    assert seen == {"key": "sk-oa", "url": "https://gateway.example/v1"}
    assert calls[0]["model"] == "gpt-5.2" and calls[0]["messages"][0]["role"] == "system"
    assert calls[1]["messages"][-1]["role"] == "tool" and calls[1]["messages"][-1]["tool_call_id"] == "c1"


def test_assistant_page_chat_flow(no_key):
    app = Path(__file__).resolve().parents[1] / "app" / "Home.py"
    at = AppTest.from_file(str(app), default_timeout=180)
    at.run()
    at.switch_page("views/7_Assistant.py").run()
    assert not at.exception, [e.value for e in at.exception]
    at.chat_input[0].set_value("top 5 customers").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.chat_message) == 2



def test_azure_endpoint_is_detected_from_any_copied_url(monkeypatch, no_key):
    monkeypatch.setenv("OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("OPENAI_MODEL", "my-gpt-deployment")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://res.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview")
    p = a.provider()
    assert (p.name, p.base_url, p.api_version, p.model) == (
        "azure", "https://res.cognitiveservices.azure.com", "2025-04-01-preview", "my-gpt-deployment")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://res.openai.azure.com/")
    assert a.provider().api_version == a.AZURE_API_VERSION     # default when the URL has none
    assert "azure-key" not in repr(a.provider())
