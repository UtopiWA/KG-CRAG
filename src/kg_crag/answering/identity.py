"""回答阶段内容寻址身份和阶段指纹。"""

from __future__ import annotations

from kg_crag.correction.identity import stable_digest
from kg_crag.models import AnswerRunIdentity, CorrectionRunResult


def build_answer_identity(
    correction: CorrectionRunResult,
    *,
    config_hash: str,
    answer_prompt_version: str,
    critic_prompt_version: str,
    checker_version: str,
    policy_version: str,
    model_revision: str,
    search_provider: str,
    search_provider_version: str,
) -> AnswerRunIdentity:
    input_payload = {
        "internal_identity": correction.identity.model_dump(mode="json"),
        "question": correction.state.question,
        "facets": sorted(
            (item.model_dump(mode="json") for item in correction.state.facets),
            key=stable_digest,
        ),
        "evidence": sorted(
            (item.model_dump(mode="json") for item in correction.state.selected_evidence),
            key=stable_digest,
        ),
        "coverage": correction.state.coverage_matrix.model_dump(mode="json")
        if correction.state.coverage_matrix
        else None,
        "sufficiency": correction.state.sufficiency.model_dump(mode="json")
        if correction.state.sufficiency
        else None,
        "stop": correction.stop.model_dump(mode="json"),
    }
    input_hash = stable_digest(input_payload)
    identity_payload = {
        "input_hash": input_hash,
        "config_hash": config_hash,
        "answer_prompt_version": answer_prompt_version,
        "critic_prompt_version": critic_prompt_version,
        "checker_version": checker_version,
        "policy_version": policy_version,
        "model_revision": model_revision,
        "search_provider": search_provider,
        "search_provider_version": search_provider_version,
    }
    return AnswerRunIdentity(
        run_id="answer-run-" + stable_digest(identity_payload)[:32],
        internal_run_id=correction.identity.run_id,
        input_hash=input_hash,
        config_hash=config_hash,
        answer_prompt_version=answer_prompt_version,
        critic_prompt_version=critic_prompt_version,
        checker_version=checker_version,
        policy_version=policy_version,
        model_revision=model_revision,
        search_provider=search_provider,
        search_provider_version=search_provider_version,
    )


def answer_state_fingerprint(payload: object) -> str:
    """为节点缓存生成稳定哈希，调用方只传可序列化结构。"""

    return stable_digest(payload)
