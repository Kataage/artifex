from __future__ import annotations

from pathlib import Path

from artifex.characters import CharacterRegistry
from artifex.config.models import RightsConfig
from artifex.db import Database
from artifex.domain import CharacterProfile, PublicationTier, ResultState
from artifex.packs import ScenePlan, VisualSpecification
from artifex.policy import (
    OperatorPolicyOutcome,
    PolicyEngine,
    RightsPolicyProfile,
    RightsPolicyRegistry,
)


def _visual() -> VisualSpecification:
    return VisualSpecification(
        composition="portrait",
        camera="eye level",
        pose="standing",
        expression="smile",
        clothing="casual",
        setting="room",
        lighting="soft",
        atmosphere="calm",
    )


def _setup(
    tmp_path: Path,
    *,
    policy: RightsPolicyProfile,
    character_policy: str | None = None,
) -> tuple[Database, PolicyEngine]:
    database = Database(f"sqlite:///{(tmp_path / 'policy.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="A",
            namespace="test",
            readiness=1,
            policy_profile=character_policy or policy.id,
        )
    )
    policies = RightsPolicyRegistry.with_packaged_defaults()
    policies.register(policy)
    engine = PolicyEngine(characters, policies, RightsConfig())
    return database, engine


def test_planner_public_intent_cannot_bypass_member_policy(tmp_path: Path) -> None:
    policy = RightsPolicyProfile(
        id="member_only",
        version="1",
        minimum_publication_tier=PublicationTier.MEMBER,
    )
    database, engine = _setup(tmp_path, policy=policy)
    scene = ScenePlan(
        ordinal=1,
        title="scene",
        purpose="test",
        character_ids=("char-a",),
        visual=_visual(),
        publication_tier=PublicationTier.PUBLIC,
    )

    decision = engine.pre_generation(
        character_ids=scene.character_ids,
        pack_format="single_feature",
        requested_tier=scene.publication_tier,
    )

    assert decision.allowed_generation is True
    assert decision.requested_tier is PublicationTier.PUBLIC
    assert decision.policy_floor is PublicationTier.MEMBER
    assert decision.publication_tier is PublicationTier.MEMBER
    assert decision.policy_refs[0].policy_id == "member_only"
    assert decision.policy_refs[0].version == "1"
    database.dispose()


def test_generation_disallowed_policy_blocks_before_generation(tmp_path: Path) -> None:
    policy = RightsPolicyProfile(
        id="blocked",
        version="2",
        allow_generation=False,
    )
    database, engine = _setup(tmp_path, policy=policy)

    decision = engine.pre_generation(
        character_ids=("char-a",),
        pack_format="single_feature",
        requested_tier=PublicationTier.PUBLIC,
    )

    assert decision.allowed_generation is False
    assert decision.publication_tier is PublicationTier.BLOCKED
    assert "generation_disallowed" in decision.reasons[0]
    database.dispose()


def test_post_generation_reclassifies_review_and_reject(tmp_path: Path) -> None:
    policy = RightsPolicyProfile(id="member_only", version="1")
    database, engine = _setup(tmp_path, policy=policy)
    pre = engine.pre_generation(
        character_ids=("char-a",),
        pack_format="single_feature",
        requested_tier=PublicationTier.PUBLIC,
    )

    review = engine.post_generation(pre, evaluation_state=ResultState.REVIEW)
    rejected = engine.post_generation(pre, evaluation_state=ResultState.REJECTED)

    assert review.publication_tier is PublicationTier.PRIVATE_REVIEW
    assert review.requires_operator_review is True
    assert rejected.publication_tier is PublicationTier.BLOCKED
    database.dispose()


def test_operator_approval_cannot_relax_rights_floor(tmp_path: Path) -> None:
    policy = RightsPolicyProfile(
        id="member_review",
        version="1",
        minimum_publication_tier=PublicationTier.MEMBER,
        requires_operator_review=True,
    )
    database, engine = _setup(tmp_path, policy=policy)
    pre = engine.pre_generation(
        character_ids=("char-a",),
        pack_format="single_feature",
        requested_tier=PublicationTier.PUBLIC,
    )
    assert pre.publication_tier is PublicationTier.PRIVATE_REVIEW

    approved = engine.apply_operator_outcome(
        pre,
        OperatorPolicyOutcome.APPROVE,
    )

    assert approved.publication_tier is PublicationTier.MEMBER
    assert approved.requires_operator_review is False
    database.dispose()


def test_unknown_policy_fails_closed(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'unknown.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="A",
            namespace="test",
            readiness=1,
            policy_profile="does-not-exist",
        )
    )
    engine = PolicyEngine(
        characters,
        RightsPolicyRegistry.with_packaged_defaults(),
        RightsConfig(fail_closed_unknown_policy=True),
    )

    decision = engine.pre_generation(
        character_ids=("char-a",),
        pack_format="single_feature",
        requested_tier=PublicationTier.PUBLIC,
    )

    assert decision.allowed_generation is False
    assert decision.publication_tier is PublicationTier.BLOCKED
    database.dispose()
