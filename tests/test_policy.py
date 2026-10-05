from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifex.characters import CharacterRegistry
from artifex.config.models import RightsConfig
from artifex.db import Database
from artifex.db.models import PackRow, SceneRow
from artifex.domain import CharacterProfile, PackState, PublicationTier, SceneState
from artifex.policy import (
    PolicyApplicationService,
    PolicyDecisionRepository,
    PolicyEngine,
    PolicyOutcome,
    PolicyProfile,
    PolicyRegistry,
    PolicyRequest,
    PolicyRule,
    UseClass,
)


def _database(tmp_path: Path) -> Database:
    database = Database(f"sqlite:///{(tmp_path / 'policy.sqlite3').as_posix()}")
    database.migrate()
    return database


def _profile(
    profile_id: str,
    *,
    public: PolicyOutcome = PolicyOutcome.ALLOW,
    member: PolicyOutcome = PolicyOutcome.ALLOW,
    commercial: PolicyOutcome = PolicyOutcome.ALLOW,
    content_rules: dict[str, PolicyRule] | None = None,
) -> PolicyProfile:
    return PolicyProfile(
        id=profile_id,
        version="1",
        use_rules={
            "private": "allow",
            "public_free": "allow",
            "paid_membership": "allow",
            "commercial": commercial,
        },
        tier_rules={
            "public": public,
            "member": member,
            "private_review": "allow",
            "blocked": "block",
        },
        content_label_rules=content_rules or {},
    )


def _engine(
    database: Database,
    characters: CharacterRegistry,
    registry: PolicyRegistry,
    *,
    default_profile: str = "allow",
) -> tuple[PolicyEngine, PolicyDecisionRepository]:
    ids = iter(f"decision-{index}" for index in range(1, 20))
    decisions = PolicyDecisionRepository(database, id_factory=lambda: next(ids))
    engine = PolicyEngine(
        registry,
        characters,
        decisions,
        RightsConfig(default_profile=default_profile),
    )
    return engine, decisions


def test_packaged_unconfigured_profile_is_conservative() -> None:
    registry = PolicyRegistry.with_packaged_defaults()
    profile = registry.require("unconfigured")

    assert profile.use_rules[UseClass.PRIVATE] is PolicyOutcome.ALLOW
    assert profile.use_rules[UseClass.PAID_MEMBERSHIP] is PolicyOutcome.REVIEW
    assert profile.tier_rules[PublicationTier.PUBLIC] is PolicyOutcome.REVIEW


def test_scene_or_model_public_intent_cannot_bypass_policy(tmp_path: Path) -> None:
    database = _database(tmp_path)
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="A",
            namespace="test",
            policy_profile="blocked-public",
        )
    )
    registry = PolicyRegistry()
    registry.register(_profile("blocked-public", public=PolicyOutcome.BLOCK))
    engine, _ = _engine(database, characters, registry)

    decision = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="scene-1",
            character_ids=("char-a",),
            use_class=UseClass.PUBLIC_FREE,
            requested_tier=PublicationTier.PUBLIC,
        )
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.effective_tier is PublicationTier.BLOCKED
    database.dispose()


def test_most_restrictive_character_policy_wins_for_duo(tmp_path: Path) -> None:
    database = _database(tmp_path)
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="A",
            namespace="test",
            policy_profile="allow",
        )
    )
    characters.upsert(
        CharacterProfile(
            id="char-b",
            display_name="B",
            namespace="test",
            policy_profile="review-member",
        )
    )
    registry = PolicyRegistry()
    registry.register(_profile("allow"))
    registry.register(
        _profile("review-member", member=PolicyOutcome.REVIEW)
    )
    engine, _ = _engine(database, characters, registry)

    decision = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="scene-duo",
            character_ids=("char-a", "char-b"),
            use_class=UseClass.PAID_MEMBERSHIP,
            requested_tier=PublicationTier.MEMBER,
        )
    )

    assert decision.outcome is PolicyOutcome.REVIEW
    assert decision.effective_tier is PublicationTier.PRIVATE_REVIEW
    assert decision.profile_versions == ("allow@1", "review-member@1")
    database.dispose()


def test_post_generation_label_can_reclassify_to_private_review(tmp_path: Path) -> None:
    database = _database(tmp_path)
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="A",
            namespace="test",
            policy_profile="label-policy",
        )
    )
    registry = PolicyRegistry()
    registry.register(
        _profile(
            "label-policy",
            content_rules={
                "needs_manual_check": PolicyRule(
                    outcome=PolicyOutcome.REVIEW,
                    force_tier=PublicationTier.PRIVATE_REVIEW,
                    reason="generated output requires manual classification",
                )
            },
        )
    )
    engine, _ = _engine(database, characters, registry)

    now = datetime.now(UTC)
    with database.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state=PackState.GENERATING.value,
                format_type="single_feature",
                payload_json={},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            SceneRow(
                id="scene-1",
                pack_id="pack-1",
                ordinal=1,
                state=SceneState.EVALUATING.value,
                publication_tier=PublicationTier.PUBLIC.value,
                payload_json={
                    "plan": {
                        "ordinal": 1,
                        "title": "Scene",
                        "purpose": "test",
                        "character_ids": ["char-a"],
                        "continuity_constraints": [],
                        "visual": {
                            "composition": "portrait",
                            "camera": "eye level",
                            "pose": "standing",
                            "expression": "smile",
                            "clothing": "casual",
                            "setting": "room",
                            "lighting": "soft",
                            "atmosphere": "calm",
                            "positive_constraints": [],
                            "negative_constraints": [],
                        },
                        "publication_tier": "public",
                        "transition_from_previous": None,
                        "series_state_updates": {},
                        "unresolved_hooks_added": [],
                        "unresolved_hooks_resolved": [],
                    }
                },
            )
        )

    service = PolicyApplicationService(database, engine)
    decision = service.evaluate_scene(
        "scene-1",
        use_class=UseClass.PUBLIC_FREE,
        content_labels=("needs_manual_check",),
        phase="post_generation",
    )

    assert decision.outcome is PolicyOutcome.REVIEW
    assert decision.effective_tier is PublicationTier.PRIVATE_REVIEW
    assert service.scenes_for_tier(PublicationTier.PRIVATE_REVIEW) == ("scene-1",)
    database.dispose()


def test_operator_can_resolve_review_but_not_hard_block(tmp_path: Path) -> None:
    database = _database(tmp_path)
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="A",
            namespace="test",
            policy_profile="review",
        )
    )
    registry = PolicyRegistry()
    registry.register(_profile("review", public=PolicyOutcome.REVIEW))
    engine, decisions = _engine(database, characters, registry)

    review = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="scene-review",
            character_ids=("char-a",),
            use_class=UseClass.PUBLIC_FREE,
            requested_tier=PublicationTier.PUBLIC,
        )
    )
    approved = decisions.resolve_review(review.decision_id, approved=True)

    assert approved.outcome is PolicyOutcome.ALLOW
    assert approved.effective_tier is PublicationTier.PUBLIC
    assert approved.review_of == review.decision_id

    registry.register(_profile("block", public=PolicyOutcome.BLOCK))
    characters.upsert(
        characters.require("char-a").model_copy(update={"policy_profile": "block"})
    )
    blocked = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="scene-block",
            character_ids=("char-a",),
            use_class=UseClass.PUBLIC_FREE,
            requested_tier=PublicationTier.PUBLIC,
        )
    )
    with pytest.raises(ValueError, match="only review"):
        decisions.resolve_review(blocked.decision_id, approved=True)
    database.dispose()


def test_missing_explicit_profile_blocks_instead_of_falling_through(tmp_path: Path) -> None:
    database = _database(tmp_path)
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="A",
            namespace="test",
            policy_profile="missing",
        )
    )
    registry = PolicyRegistry()
    registry.register(_profile("allow"))
    engine, _ = _engine(database, characters, registry)

    decision = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="scene-1",
            character_ids=("char-a",),
            use_class=UseClass.PUBLIC_FREE,
            requested_tier=PublicationTier.PUBLIC,
        )
    )

    assert decision.outcome is PolicyOutcome.BLOCK
    assert "unavailable" in decision.reasons[0]
    database.dispose()
