from __future__ import annotations

from pathlib import Path

from artifex.characters import CharacterRegistry
from artifex.config.models import RightsConfig
from artifex.db import Database
from artifex.domain import CharacterProfile, PublicationTier, ResultState
from artifex.policy import (
    OperatorPolicyOutcome,
    PolicyDecisionRepository,
    PolicyEngine,
    PolicyGateService,
    RightsPolicyProfile,
    RightsPolicyRegistry,
)


def test_policy_decision_chain_persists_versions_and_operator_outcome(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'policy-chain.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="A",
            namespace="test",
            readiness=1,
            policy_profile="member_review",
        )
    )
    policies = RightsPolicyRegistry.with_packaged_defaults()
    policies.register(
        RightsPolicyProfile(
            id="member_review",
            version="2026.10",
            minimum_publication_tier=PublicationTier.MEMBER,
            requires_operator_review=True,
        )
    )

    ids = iter(("decision-1", "decision-2", "decision-3"))
    repository = PolicyDecisionRepository(database, id_factory=lambda: next(ids))
    service = PolicyGateService(
        PolicyEngine(characters, policies, RightsConfig()),
        repository,
    )

    pre = service.pre_generation(
        subject_type="scene",
        subject_id="scene-1",
        character_ids=("char-a",),
        pack_format="single_feature",
        requested_tier=PublicationTier.PUBLIC,
    )
    post = service.post_generation(
        pre,
        subject_type="scene",
        subject_id="scene-1",
        evaluation_state=ResultState.REVIEW,
    )
    final = service.operator_outcome(
        post,
        subject_type="scene",
        subject_id="scene-1",
        outcome=OperatorPolicyOutcome.APPROVE,
    )

    assert final.decision.publication_tier is PublicationTier.MEMBER

    pre_row = repository.require("decision-1")
    post_row = repository.require("decision-2")
    final_row = repository.require("decision-3")

    refs = pre_row.payload_json["decision"]["policy_refs"]
    assert refs == [{"policy_id": "member_review", "version": "2026.10"}]
    assert post_row.payload_json["parent_decision_id"] == "decision-1"
    assert final_row.payload_json["parent_decision_id"] == "decision-2"
    assert final_row.payload_json["operator_outcome"] == "approve"
    assert final_row.policy_version == "1"
    database.dispose()
