from __future__ import annotations

import argparse
import json

from backend.src.slothbearflow_backend.config import get_settings
from backend.src.slothbearflow_backend.evaluation import (
    ProductionAclEvaluationStore,
    evaluation_gate_passed,
    evaluate_rag_dataset,
    load_dataset,
)
from backend.src.slothbearflow_backend.rag.milvus_store import get_vector_store
from backend.src.slothbearflow_backend.rag.security import RagAccessContext
from backend.src.slothbearflow_backend.tools.rag_tool import retrieve_knowledge_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the versioned RAG regression set.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--minimum-pass-rate", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_pass_rate <= 1.0:
        parser.error("--minimum-pass-rate must be between 0 and 1")
    settings = get_settings()
    vector_store = get_vector_store(settings)
    if vector_store is None:
        raise SystemExit("RAG vector store is unavailable.")
    cases = load_dataset(args.dataset) if args.dataset else load_dataset()
    acl_store = ProductionAclEvaluationStore()

    def retrieve_case(case):
        store = acl_store if "acl-fixture" in case.tags else vector_store
        return retrieve_knowledge_context(
            store,
            case.question,
            settings=settings,
            access_context=RagAccessContext(
                tenant_id=case.tenant_id,
                user_id=case.user_id,
                roles=set(case.roles),
                allow_legacy=case.allow_legacy,
            ),
        )

    report = evaluate_rag_dataset(
        cases,
        lambda query: retrieve_knowledge_context(
            vector_store,
            query,
            settings=settings,
            access_context=RagAccessContext(
                tenant_id=settings.auth_local_tenant_id,
                user_id=settings.auth_local_user_id,
                roles=set(settings.auth_local_roles_json),
                allow_legacy=settings.rag_allow_legacy_documents,
            ),
        ),
        retrieve_case=retrieve_case,
    )
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    if not evaluation_gate_passed(
        report,
        minimum_pass_rate=args.minimum_pass_rate,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
