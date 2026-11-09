"""不連網、不呼叫 Gemini 的測試：證交所資料和模型回覆都用假的。"""
import sys
from pathlib import Path

import httpx
import pytest
from google.genai import types
from streamlit.testing.v1 import AppTest

COACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COACH))

import agent  # noqa: E402
import tools  # noqa: E402

# 市價 104.10／淨值 103.76 是 11/2 實際跑出來的資料，p* 應該和 1102 文件一致
QUOTE = {"0050": {"name": "元大台灣50", "price": 104.10, "nav": 103.76, "nav_type": "預估", "time": "20261102 13:30:45"},
         "00679B": {"name": "元大美債20年", "price": 30.0, "nav": 28.0, "nav_type": "預估", "time": "20261102 13:30:45"}}


@pytest.fixture(autouse=True)
def fake_twse(monkeypatch):
    monkeypatch.setattr(tools, "fetch_twse", lambda: QUOTE)


def reply(part):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))])


def call(name, **args):
    return reply(types.Part(function_call=types.FunctionCall(name=name, args=args)))


def answer(text):
    return reply(types.Part(text=text))


class FakeClient:
    """照順序吐出事先排好的模型回覆，並記下每次收到的 contents。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.seen.append(list(contents))
        return self.responses.pop(0)


def test_check_premium():
    r = tools.check_premium("0050")
    assert r["premium_pct"] == 0.33 and r["status"] == "正常範圍"


def test_check_premium_high_warns():
    assert "別追價" in tools.check_premium("00679B")["status"]


def test_compare_matches_1102_doc():
    r = tools.compare_channels("0050", 3000, 3)
    assert r["break_even_premium_pct"] == 1.085
    assert r["cheaper_channel"] == "ETF（證券戶）"
    assert r["saving_twd"] == 23


def test_compare_explains_itself():
    r = tools.compare_channels("0050", 3000, 3)
    assert "不是經理費" in r["feeder_drag_meaning"]
    assert "短期偏向連結基金" in r["principle"]
    assert "1.470%" in r["reason"] and "ETF（證券戶）" in r["reason"]


@pytest.mark.parametrize("lev,drop", [(1.5, 0.5667), (2, 0.35), (2.5, 0.22)])
def test_margin_call_drop(lev, drop):
    assert tools.margin_call_drop(lev) == pytest.approx(drop, abs=1e-4)


def test_touch_probability_matches_hand_calc():
    # 2Φ(ln 0.65 / 0.3229)：ln 0.65 = −0.4308，z = −1.334，Φ = 0.0911
    assert tools.touch_probability(0.35, 0.3229) == pytest.approx(0.182, abs=0.001)


def fake_closes(monkeypatch):
    series = [("d", 100 * (1.01 if i % 2 else 1.0)) for i in range(40)]
    monkeypatch.setattr(tools, "fetch_closes", lambda code: series)


def test_margin_risk_result(monkeypatch):
    fake_closes(monkeypatch)
    r = tools.margin_risk("0050", 2)
    assert r["margin_call_drop_pct"] == 35.0
    assert [x["leverage"] for x in r["comparison"]] == [1.5, 2.0, 2.5]
    assert r["limits"] and r["ask_user"]


def test_margin_risk_custom_leverage_in_comparison(monkeypatch):
    fake_closes(monkeypatch)
    assert 1.8 in [x["leverage"] for x in tools.margin_risk("0050", 1.8)["comparison"]]


@pytest.mark.parametrize("lev", [1, 0.5, 3])
def test_margin_risk_bad_leverage(monkeypatch, lev):
    fake_closes(monkeypatch)
    assert "error" in agent.run_tool("margin_risk", {"code": "0050", "leverage": lev})


def test_margin_risk_too_little_history(monkeypatch):
    monkeypatch.setattr(tools, "fetch_closes", lambda code: [("d", 100.0)] * 5)
    assert "error" in agent.run_tool("margin_risk", {"code": "9999", "leverage": 2})


@pytest.mark.parametrize("text,value", [("3,000 元", 3000), ("3千元", 3000), ("三千元", 3000), ("一萬五", 15000),
                                        ("２倍", 2), ("兩倍", 2), ("放半年", 0.5), ("一年半", 1.5), ("2.5 年半", 3.0)])
def test_numbers_said(text, value):
    assert any(abs(value - n) < 1e-6 for n in agent.numbers_said(text))


def test_six_months_means_half_year():
    assert not agent.invented_args("compare_channels", {"code": "0050", "amount": 3000, "years": 0.5},
                                   ["0050 3000 元放 6 個月"])


@pytest.mark.parametrize("args,bad", [({"code": "0050", "leverage": 2}, ["leverage"]),
                                      ({"code": "00878", "leverage": 2}, ["code", "leverage"])])
def test_guardrail_flags_invented_args(args, bad):
    assert agent.invented_args("margin_risk", args, ["我想開槓桿買 0050，風險大嗎？"]) == bad


def test_guardrail_accepts_said_args():
    assert not agent.invented_args("margin_risk", {"code": "0050", "leverage": 2}, ["0050 風險", "開 2 倍"])
    assert not agent.invented_args("check_premium", {"code": "0050"}, ["元大台灣50 有溢價嗎"])
    assert not agent.invented_args("compare_channels", {"code": "0050", "amount": 3000, "years": 3, "discount": 0.28},
                                   ["0050 3000 元 3 年，手續費 28 折"])


def test_guardrail_blocks_before_tool_runs(monkeypatch):
    fake_closes(monkeypatch)
    c = FakeClient(call("margin_risk", code="0050", leverage=2), answer("請問幾倍？"))
    text, trace, _ = agent.run_turn(c, "m", [], "我想開槓桿買 0050，風險大嗎？")
    assert text == "請問幾倍？" and trace[0]["blocked"] is True
    assert "護欄" in trace[0]["result"]["error"]


def test_guardrail_uses_earlier_turns(monkeypatch):
    fake_closes(monkeypatch)
    c = FakeClient(answer("幾倍？"), call("margin_risk", code="0050", leverage=2), answer("好"))
    _, _, h = agent.run_turn(c, "m", [], "我想開槓桿買 0050")
    _, trace, _ = agent.run_turn(c, "m", h, "2 倍")
    assert trace[0]["blocked"] is False and trace[0]["result"]["margin_call_drop_pct"] == 35.0


def test_guardrail_handles_number_as_text():
    assert not agent.invented_args("margin_risk", {"code": "0050", "leverage": "2"}, ["0050 開 2 倍"])
    assert agent.invented_args("margin_risk", {"code": "0050", "leverage": "很多"}, ["0050 開 2 倍"]) == ["leverage"]


def test_five_year_gap_matches_1102_doc():
    assert tools.feeder_drag(5, "actual") == pytest.approx(0.0138)


def test_split_jump_is_not_counted_as_volatility(monkeypatch):
    smooth = [("d", 100 * (1.01 if i % 2 else 1.0)) for i in range(40)]
    split = smooth[:20] + [("d", p / 4) for _, p in smooth[20:]]
    monkeypatch.setattr(tools, "fetch_closes", lambda code: smooth)
    base = tools.margin_risk("0050", 2)["daily_volatility_pct"]
    monkeypatch.setattr(tools, "fetch_closes", lambda code: split)
    assert tools.margin_risk("0050", 2)["daily_volatility_pct"] == pytest.approx(base, abs=0.05)


def test_partial_history_is_not_cached(monkeypatch):
    monkeypatch.setattr(tools, "_get_json", lambda url: {"data": []})
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    tools._closes_cache.clear()
    assert tools.fetch_closes("0050") == [] and not tools._closes_cache


def test_market_note():
    from datetime import datetime
    assert tools.market_note(datetime(2026, 11, 9, 10, 0)) is None
    assert tools.market_note(datetime(2026, 11, 9, 18, 0))
    assert tools.market_note(datetime(2026, 11, 8, 10, 0))  # 星期日


def test_negative_min_fee_is_error():
    assert "error" in agent.run_tool("compare_channels", {"code": "0050", "amount": 3000, "years": 3, "min_fee": -1})


def test_small_amount_hits_min_fee():
    # 500 元手續費 0.71 元，低於最低 1 元，要用最低手續費算：(2×1)/500 + 0.1% = 0.5%
    assert tools.compare_channels("0050", 500, 1)["etf_round_trip_cost_pct"] == 0.5


def test_no_feeder_fund():
    assert tools.compare_channels("00679B", 3000, 3)["has_feeder_fund"] is False


def test_unsupported_feeder():
    assert tools.compare_channels("0056", 3000, 3)["supported"] is False


@pytest.mark.parametrize("kw", [{"amount": 0, "years": 3}, {"amount": 3000, "years": -1},
                                {"amount": 3000, "years": 3, "discount": 2}])
def test_bad_args_become_error(kw):
    assert "error" in agent.run_tool("compare_channels", {"code": "0050", **kw})


def test_unknown_code_becomes_error():
    assert "error" in agent.run_tool("check_premium", {"code": "9999"})


def test_unknown_tool_becomes_error():
    assert "error" in agent.run_tool("buy_stock", {})


def test_network_failure_becomes_error(monkeypatch):
    def boom():
        raise TimeoutError("timed out")
    monkeypatch.setattr(tools, "fetch_twse", boom)
    assert "timed out" in agent.run_tool("check_premium", {"code": "0050"})["error"]


def test_tool_loop_sends_result_back():
    c = FakeClient(call("check_premium", code="0050"), answer("溢價 0.33%"))
    text, trace, history = agent.run_turn(c, "m", [], "0050 有溢價嗎")
    assert text == "溢價 0.33%"
    assert [t["type"] for t in trace] == ["tool", "answer"]
    sent = c.seen[1]
    assert sent[1].parts[0].function_call.name == "check_premium"
    assert sent[2].parts[0].function_response.response["premium_pct"] == 0.33
    assert history[-1].role == "model"


def test_tokens_recorded_per_model_call():
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5, total_token_count=125)
    first = call("check_premium", code="0050")
    first.usage_metadata = usage
    second = answer("溢價 0.33%")
    second.usage_metadata = usage
    _, trace, _ = agent.run_turn(FakeClient(first, second), "m", [], "問")
    assert [t["tokens"]["total"] for t in trace] == [125, 125]
    assert trace[1]["tokens"] == {"input": 100, "output": 20, "thinking": 5, "total": 125}


def test_parallel_tool_calls_count_tokens_once():
    parts = [types.Part(function_call=types.FunctionCall(name="check_premium", args={"code": "0050"})),
             types.Part(function_call=types.FunctionCall(name="check_premium", args={"code": "00679B"}))]
    both = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=50))
    _, trace, _ = agent.run_turn(FakeClient(both, answer("好")), "m", [], "問")
    assert trace[0]["tokens"]["total"] == 50 and trace[1]["tokens"] is None


def test_missing_usage_is_zero():
    _, trace, _ = agent.run_turn(FakeClient(answer("好")), "m", [], "問")
    assert trace[0]["tokens"]["total"] == 0


def test_no_tool_needed():
    text, trace, _ = agent.run_turn(FakeClient(answer("請問金額多少？")), "m", [], "我想比較通路")
    assert text == "請問金額多少？" and trace[0]["type"] == "answer"


def test_history_carries_over():
    c = FakeClient(answer("第一次"), answer("第二次"))
    _, _, h = agent.run_turn(c, "m", [], "一")
    agent.run_turn(c, "m", h, "二")
    assert [x.parts[0].text for x in c.seen[1]] == ["一", "第一次", "二"]


def test_stops_after_max_steps():
    c = FakeClient(*[call("check_premium", code="0050") for _ in range(agent.MAX_STEPS)])
    text, trace, history = agent.run_turn(c, "m", [], "問")
    assert "太多次" in text and len(trace) == agent.MAX_STEPS and history == []


def test_app_without_key_shows_warning(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(str(COACH / "app.py")).run()
    assert not at.exception and "GEMINI_API_KEY" in at.warning[0].value


def test_app_connection_error_shows_message(monkeypatch):
    class Broken:
        models = None

        def __init__(self):
            self.models = self

        def generate_content(self, model, contents, config):
            raise httpx.ReadError("connection reset")

    monkeypatch.setattr(agent, "make_client", lambda: Broken())
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(str(COACH / "app.py")).run()
    at.chat_input[0].set_value("0050 有溢價嗎").run()
    assert not at.exception
    assert "再送一次" in at.chat_message[-1].markdown[0].value


def test_client_has_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    assert agent.make_client()._api_client._http_options.timeout == agent.REQUEST_TIMEOUT_MS


def test_app_full_turn(monkeypatch):
    fake = FakeClient(call("check_premium", code="0050"), answer("溢價 0.33%"))
    monkeypatch.setattr(agent, "make_client", lambda: fake)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(str(COACH / "app.py")).run()
    at.chat_input[0].set_value("0050 有溢價嗎").run()
    assert not at.exception
    assert at.chat_message[-1].markdown[0].value == "溢價 0.33%"
