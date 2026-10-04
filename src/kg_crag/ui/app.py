"""Streamlit 单页演示；本模块不导入任何领域工作流。"""

from __future__ import annotations

import sys
from typing import Any

import httpx
import streamlit as st  # type: ignore[import-not-found]

from kg_crag.settings import get_settings

_CASES = {
    "内部证据充分": "sufficient",
    "纠错补全缺失 facet": "corrected-gap",
    "依赖不可用时保守停止": "conservative-stop",
}


def _post_query(
    base_url: str,
    *,
    question: str,
    mode: str,
    replay_case_id: str | None,
) -> dict[str, Any]:
    """只通过版本化 API 发起单问题请求，并约束客户端超时。"""

    payload: dict[str, Any] = {"question": question, "mode": mode}
    if replay_case_id is not None:
        payload["replay_case_id"] = replay_case_id
    with httpx.Client(base_url=base_url, timeout=35.0) as client:
        response = client.post("/v1/queries", json=payload)
        data = response.json()
        if response.is_error:
            error = data.get("error", {}) if isinstance(data, dict) else {}
            message = error.get("message", "API 返回未知错误。")
            request_id = data.get("request_id", "unknown") if isinstance(data, dict) else "unknown"
            retryable = bool(error.get("retryable", False))
            raise RuntimeError(f"{message} (request_id={request_id}，可重试={retryable})")
        if not isinstance(data, dict):
            raise RuntimeError("API 响应不是有效对象。")
        return data


def _render_result(payload: dict[str, Any]) -> None:
    """按公共响应字段展示证据、facet、动作与预算，不推断领域结论。"""

    if payload.get("mode") == "replay":
        st.warning(payload.get("replay_notice", "固定回放，非实时结果。"))
        st.caption(f"回放版本: {payload.get('replay_fixture_version', 'unknown')}")
    st.subheader("回答")
    st.write(payload.get("answer", ""))
    st.caption(f"停止原因: {payload.get('stop_reason', 'unknown')}")

    st.subheader("引用与来源")
    citations = payload.get("citations", [])
    if not citations:
        st.info("当前结果没有可发布引用。")
    for item in citations:
        channels = " / ".join(item.get("source_channels", []))
        location = " · ".join(
            str(value)
            for value in (item.get("paper_id"), item.get("section"), item.get("page"))
            if value is not None
        )
        st.markdown(
            f"- **{item.get('citation_id')}** [{channels}] {item.get('title')} — {location}"
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Facet 覆盖")
        for item in payload.get("facets", []):
            st.write(f"{item.get('status')} · {item.get('label')} ({item.get('facet_id')})")
    with right:
        st.subheader("路由与纠错")
        st.write(" → ".join(payload.get("retrieval_path", [])) or "无")
        for item in payload.get("actions", []):
            st.write(
                f"#{item.get('sequence')} {item.get('action')} · "
                f"{item.get('reason')} · {item.get('status')}"
            )

    st.subheader("预算摘要")
    st.json(payload.get("budget", {}), expanded=False)
    st.caption(f"trace_id={payload.get('trace_id', 'unknown')}")


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="KG-CRAG 科研问答演示", layout="wide")
    st.title("证据自适应科研问答 Agent")
    st.caption("Knowledge-Gap-Aware Corrective Retrieval Agent · Graph 为可选工具")

    mode_label = st.radio("运行模式", ["固定回放", "实时工作流"], horizontal=True)
    mode = "replay" if mode_label == "固定回放" else "live"
    selected_case = st.selectbox("固定案例", list(_CASES)) if mode == "replay" else None
    default_question = (
        "请展示这个案例的证据覆盖与停止原因。" if mode == "replay" else "请输入一个科研文献问题。"
    )
    question = st.text_area("问题", value=default_question, max_chars=2000)
    if st.button("运行", type="primary"):
        try:
            result = _post_query(
                settings.ui_api_base_url,
                question=question,
                mode=mode,
                replay_case_id=_CASES[selected_case] if selected_case else None,
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            _render_result(result)


def run() -> None:
    """以安装后的命令启动 Streamlit，避免要求用户记住模块路径。"""

    from streamlit.web import cli as streamlit_cli  # type: ignore[import-not-found]

    sys.argv = ["streamlit", "run", __file__, "--server.address=0.0.0.0"]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
