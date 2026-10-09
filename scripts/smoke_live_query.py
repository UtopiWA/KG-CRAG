"""对一个非评测问题执行低 Token 实时链路烟雾测试。"""

from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from pathlib import Path

from pydantic import ValidationError

from kg_crag.application.live import build_live_query_runtime
from kg_crag.errors import KGCRAGError
from kg_crag.models import ApplicationMode, QueryRequest
from kg_crag.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        default="How does CAMEL use role-playing to facilitate cooperation among chat agents?",
    )
    parser.add_argument(
        "--confirm-online",
        action="store_true",
        help="确认允许调用当前 .env 中的 LLM；不会执行正式评测",
    )
    parser.add_argument(
        "--allow-web",
        action="store_true",
        help="内部证据不足时允许使用 .env 配置的单次受控 Web 搜索",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    if not args.confirm_online:
        raise ValueError("online smoke requires --confirm-online")
    request = QueryRequest(
        question=args.question,
        mode=ApplicationMode.LIVE,
        allow_web=args.allow_web,
        include_trace=True,
    )
    # 烟雾测试只验证链路: 关闭 Critic、重排和 Graph；Web 必须由命令行显式允许。
    settings = Settings(
        enable_live_query=True,
        live_enable_reranker=False,
        live_enable_answer_critic=False,
        live_enable_graph=False,
        enable_web_fallback=args.allow_web,
    )
    runtime = await build_live_query_runtime(settings, workspace_root=PROJECT_ROOT)
    execution = await runtime.run(request)
    used = execution.answer.budget.answer.used
    generation_events = [
        {
            "event": item.event,
            "failure_category": item.details.get("failure_category"),
            "retryable": item.details.get("retryable"),
            "provider_status": item.details.get("provider_status"),
            "provider_error_type": item.details.get("provider_error_type"),
            "provider_finish_reason": item.details.get("provider_finish_reason"),
            "provider_analysis_chars": item.details.get("provider_analysis_chars"),
            "usage_estimated": item.details.get("usage_estimated"),
        }
        for item in execution.answer.trace
        if item.event
        in {
            "answer_generated",
            "answer_generation_failed",
            "answer_generation_retry_scheduled",
        }
    ]
    decision_actions = [
        item.details.get("action")
        for item in execution.answer.trace
        if item.event == "reflection_decided"
    ]
    payload = {
        "answer": execution.answer.answer,
        "stop_reason": execution.answer.stop_reason.value,
        "citations": len(execution.answer.citations),
        "external_evidence": len(execution.answer.external_evidence),
        "web_calls": used.web_calls,
        "model_calls": used.answer_calls + used.critic_calls,
        "accounted_input_tokens": used.input_tokens,
        "accounted_output_tokens": used.output_tokens,
        "usage_estimated": any(
            bool(item.details.get("usage_estimated"))
            for item in execution.answer.trace
            if item.event in {"answer_generated", "answer_generation_failed"}
        ),
        "retrieval_path": [item.value for item in execution.answer.retrieval_path],
        "trace_events": len(execution.answer.trace),
        "generation_events": generation_events,
        "finding_codes": (
            [item.code.value for item in execution.answer.evaluation.findings]
            if execution.answer.evaluation is not None
            else []
        ),
        "decision_actions": decision_actions,
        "runtime_identity": execution.runtime_identity.model_dump(mode="json"),
        "formal_evaluation": False,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    try:
        return asyncio.run(run(parse_args()))
    except ValidationError as exc:
        fields = [
            {"field": ".".join(str(item) for item in error["loc"]), "type": error["type"]}
            for error in exc.errors(include_input=False, include_url=False)
        ]
        stack = [
            f"{Path(frame.filename).name}:{frame.name}:{frame.lineno}"
            for frame in traceback.extract_tb(exc.__traceback__)[-6:]
        ]
        print(
            json.dumps(
                {
                    "error": "ValidationError",
                    "fields": fields,
                    "stack": stack,
                    "formal_evaluation": False,
                }
            )
        )
        return 1
    except (KGCRAGError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "formal_evaluation": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
