"""ETF 定期定額行為教練：Streamlit 聊天介面。執行：streamlit run coach/app.py"""
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors

import agent

load_dotenv(Path(__file__).parent / ".env")  # 不管從哪個資料夾啟動都讀 coach/.env

st.set_page_config(page_title="ETF 行為教練", page_icon="📈")
st.title("ETF 定期定額行為教練")
st.caption("只提醒成本和風險，決定權在你。僅供課程學習，非投資建議；免費額度的對話可能被 Google 用來改進產品，請不要輸入個人資料或帳戶資訊。")

client = agent.make_client()
if client is None:
    st.warning("找不到 GEMINI_API_KEY。請到 Google AI Studio 申請，再把 coach/.env.example 複製成 coach/.env 填進去。")
    st.stop()

st.session_state.setdefault("history", [])
st.session_state.setdefault("messages", [])


def show_trace(trace):
    """把 agent 每一步做了什麼攤開來看（可觀測性）。"""
    with st.expander("執行紀錄"):
        for step in trace:
            used = step.get("tokens")
            note = f"，輸入 {used['input']}／輸出 {used['output']}／思考 {used['thinking']} token" if used else ""
            if step["type"] == "tool":
                title = "護欄擋下工具" if step.get("blocked") else "呼叫工具"
                st.markdown(f"**{title} `{step['name']}`**（模型思考 {step['seconds']} 秒{note}）")
                st.json({"參數": step["args"], "結果": step["result"]})
            else:
                st.markdown(f"**產生回答**（{step['seconds']} 秒{note}）")
        st.markdown(f"**這輪總共 {sum((s.get('tokens') or {}).get('total', 0) for s in trace)} token**")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("trace"):
            show_trace(m["trace"])

if prompt := st.chat_input("例如：0050 現在有溢價嗎？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("教練查資料中…"):
                text, trace, st.session_state.history = agent.run_turn(
                    client, agent.current_model(), st.session_state.history, prompt)
        except errors.APIError as e:
            text, trace = (f"呼叫 Gemini 失敗（{e.code}）：{e.message}。"
                           "429 代表免費額度用完或太頻繁；503、504 代表 Gemini 太忙。等一下再送一次。"), []
        except httpx.TransportError as e:
            text, trace = f"連線到 Gemini 時中斷或逾時（{type(e).__name__}），這題沒有回答，請再送一次。", []
        st.markdown(text)
        if trace:
            show_trace(trace)
    st.session_state.messages.append({"role": "assistant", "content": text, "trace": trace})
