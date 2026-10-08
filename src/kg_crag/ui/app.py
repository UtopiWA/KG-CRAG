"""只消费公共 HTTP API 的 Gradio 单轮科研问答界面。"""

from __future__ import annotations

import html
import json
import os
from collections.abc import MutableMapping
from typing import Any, Protocol, cast

import httpx

from kg_crag.settings import get_settings

_CASES = {
    "内部证据充分": "sufficient",
    "纠错补全缺失 facet": "corrected-gap",
    "依赖不可用时保守停止": "conservative-stop",
}
_EXAMPLES = [
    "CAMEL 如何通过角色扮演促进多个智能体协作?",
    "Voyager 的技能库如何支持持续学习?",
    "AgentBench 主要评估智能体的哪些能力?",
]
_LOCAL_PROXY_BYPASS = ("localhost", "127.0.0.1", "::1")


def _ensure_localhost_proxy_bypass(
    environment: MutableMapping[str, str] | None = None,
) -> None:
    """保留已有代理例外，并确保 Gradio 与本地 API 自检不经过系统代理。"""

    target = os.environ if environment is None else environment
    entries: list[str] = []
    seen: set[str] = set()
    for key in ("NO_PROXY", "no_proxy"):
        for raw in target.get(key, "").split(","):
            value = raw.strip()
            normalized = value.casefold()
            if value and normalized not in seen:
                entries.append(value)
                seen.add(normalized)
    for host in _LOCAL_PROXY_BYPASS:
        if host.casefold() not in seen:
            entries.append(host)
            seen.add(host.casefold())
    combined = ",".join(entries)
    # 不同代理库读取的变量大小写可能不同，因此仅在当前 UI 进程内同步两种写法。
    target["NO_PROXY"] = combined
    target["no_proxy"] = combined


def _post_query(
    base_url: str,
    *,
    question: str,
    mode: str,
    replay_case_id: str | None,
    allow_web: bool,
    include_trace: bool,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """每轮只提交当前问题；历史记录绝不进入 API 请求。"""

    payload: dict[str, Any] = {
        "question": question,
        "mode": mode,
        "allow_web": allow_web if mode == "live" else False,
        "include_trace": include_trace,
    }
    if replay_case_id is not None and mode == "replay":
        payload["replay_case_id"] = replay_case_id
    with httpx.Client(base_url=base_url, timeout=125.0, transport=transport) as client:
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
        if include_trace and isinstance(data.get("trace_id"), str):
            trace = client.get(f"/v1/traces/{data['trace_id']}")
            if not trace.is_error and isinstance(trace.json(), dict):
                data["trace"] = trace.json()
        return data


def _diagnostics(payload: dict[str, Any]) -> str:
    """生成有界折叠诊断，不把内部字段或原始 Provider 响应带入页面。"""

    summary = {
        "运行模式": payload.get("mode"),
        "停止原因": payload.get("stop_reason"),
        "检索路径": payload.get("retrieval_path", []),
        "引用": payload.get("citations", []),
        "Facet": payload.get("facets", []),
        "纠错动作": payload.get("actions", []),
        "预算": payload.get("budget", {}),
        "运行身份": payload.get("runtime_identity"),
        "Trace": payload.get("trace"),
        "trace_id": payload.get("trace_id"),
    }
    rendered = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    return f"<details><summary>查看引用、纠错与资源诊断</summary><pre>{rendered}</pre></details>"


def _assistant_message(payload: dict[str, Any]) -> str:
    label = "实时结果" if payload.get("mode") == "live" else "固定回放(非正式实验结果)"
    notice = payload.get("replay_notice")
    header = f"**{label}**"
    if notice:
        header += f"\n\n> {notice}"
    return f"{header}\n\n{payload.get('answer', '')}\n\n{_diagnostics(payload)}"


def submit_message(
    message: str,
    history: list[dict[str, str]] | None,
    mode_label: str,
    case_label: str,
    allow_web: bool,
    include_trace: bool,
) -> tuple[str, list[dict[str, str]]]:
    """Gradio 事件函数保持无领域逻辑，错误仅显示安全公共信封。"""

    normalized = " ".join(message.split())
    current = list(history or [])
    if not normalized:
        return "", current
    current.append({"role": "user", "content": normalized})
    mode = "live" if mode_label == "实时工作流" else "replay"
    try:
        payload = _post_query(
            get_settings().ui_api_base_url,
            question=normalized,
            mode=mode,
            replay_case_id=_CASES.get(case_label) if mode == "replay" else None,
            allow_web=allow_web,
            include_trace=include_trace,
        )
        content = _assistant_message(payload)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        content = f"**请求未完成**\n\n{exc}"
    current.append({"role": "assistant", "content": content})
    return "", current


def clear_chat() -> tuple[str, list[dict[str, str]]]:
    return "", []


class _Demo(Protocol):
    def launch(self, **kwargs: object) -> object: ...


def build_demo() -> _Demo:
    """延迟导入 Gradio，使未安装 demo 依赖时核心包仍可使用。"""

    import gradio as gr

    with gr.Blocks(title="KG-CRAG 科研问答") as demo:
        gr.Markdown(
            "# 证据自适应科研问答 Agent\n"
            "每轮问题独立处理；历史仅用于页面展示，Graph 与 Web 均为可选工具。"
        )
        with gr.Row():
            mode = gr.Radio(["实时工作流", "固定回放"], value="实时工作流", label="运行模式")
            case = gr.Dropdown(list(_CASES), value=next(iter(_CASES)), label="回放案例")
            allow_web = gr.Checkbox(False, label="内部证据不足时允许 Web")
            include_trace = gr.Checkbox(True, label="显示 Trace")
        chatbot = gr.Chatbot(type="messages", height=560, label="对话")
        message = gr.Textbox(
            placeholder="输入任意科研文献问题，按 Enter 发送",
            lines=2,
            max_lines=6,
            label="问题",
        )
        gr.Examples(_EXAMPLES, inputs=message, label="示例问题")
        with gr.Row():
            send = gr.Button("发送", variant="primary")
            stop = gr.Button("停止")
            clear = gr.Button("清空")
        inputs = [message, chatbot, mode, case, allow_web, include_trace]
        event = send.click(submit_message, inputs=inputs, outputs=[message, chatbot])
        submit_event = message.submit(submit_message, inputs=inputs, outputs=[message, chatbot])
        stop.click(fn=None, cancels=[event, submit_event])
        clear.click(clear_chat, outputs=[message, chatbot], cancels=[event, submit_event])
    return cast(_Demo, demo)


def run() -> None:
    _ensure_localhost_proxy_bypass()
    settings = get_settings()
    build_demo().launch(
        server_name=settings.ui_host,
        server_port=settings.ui_port,
        share=False,
        show_error=False,
    )


if __name__ == "__main__":
    run()
