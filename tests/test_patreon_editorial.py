from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from artifex.archive import PackArchive
from artifex.characters import CharacterRegistry
from artifex.config.models import PatreonConfig, RightsConfig
from artifex.db import Database
from artifex.db.models import GenerationAttemptRow, PackRow, SceneRow
from artifex.domain import (
    CharacterProfile,
    ContentRating,
    PackState,
    PublicationTier,
    SceneState,
)
from artifex.packs import (
    ContentPackPlan,
    ScenePlan,
    VisualSpecification,
)
from artifex.planner.models import PackFormat
from artifex.policy import (
    PolicyApplicationService,
    PolicyDecisionRepository,
    PolicyEngine,
    PolicyOutcome,
    PolicyProfile,
    PolicyRegistry,
    PolicyRequest,
    UseClass,
)


def _visual() -> VisualSpecification:
    return VisualSpecification(
        composition="full body",
        camera="eye level",
        pose="standing",
        expression="smile",
        clothing="character outfit",
        setting="studio",
        lighting="soft light",
        atmosphere="clean",
    )


def _scene(
    ordinal: int,
    characters: tuple[str, ...],
    role: str,
    *,
    tier: str = "public",
    rating: str = "general",
) -> ScenePlan:
    return ScenePlan(
        ordinal=ordinal,
        title=f"Scene {ordinal}",
        purpose=f"{role} purpose",
        character_ids=characters,
        role=role,
        visual=_visual(),
        publication_tier=tier,
        planned_content_rating=rating,
    )


_VALID_FORMAT_ROLES: dict[str, tuple[str, ...]] = {
    "single_feature": ("feature", "detail"),
    "continuation": ("continuation", "resolution"),
    "mini_story": ("opening", "development", "resolution"),
    "variation_pack": ("variation", "alternate"),
    "outfit_feature": ("outfit_reveal", "detail"),
    "seasonal": ("seasonal_hero", "detail"),
    "trend": ("trend_hero", "detail"),
    "evergreen": ("feature", "detail"),
    "experimental": ("experiment",),
    "duo": ("interaction", "detail"),
    "group": ("interaction", "detail"),
}


def _characters_for(format_name: str) -> tuple[str, ...]:
    if format_name == "duo":
        return ("a", "b")
    if format_name == "group":
        return ("a", "b", "c")
    return ("a",)


@pytest.mark.parametrize("format_name", tuple(_VALID_FORMAT_ROLES))
def test_every_pack_format_has_a_valid_scene_role_contract(format_name: str) -> None:
    characters = _characters_for(format_name)
    roles = _VALID_FORMAT_ROLES[format_name]

    plan = ContentPackPlan(
        format=format_name,
        editorial_archetype="public_only",
        character_ids=characters,
        title=f"{format_name} test",
        logline="Structured format validation.",
        scenes=tuple(
            _scene(index, characters, role)
            for index, role in enumerate(roles, start=1)
        ),
    )

    assert plan.format is PackFormat(format_name)


@pytest.mark.parametrize("format_name", tuple(_VALID_FORMAT_ROLES))
def test_every_pack_format_rejects_an_invalid_scene_role(format_name: str) -> None:
    characters = _characters_for(format_name)
    roles = list(_VALID_FORMAT_ROLES[format_name])
    roles[0] = "feature" if format_name == "experimental" else "experiment"

    with pytest.raises(ValueError, match="role|mini_story"):
        ContentPackPlan(
            format=format_name,
            editorial_archetype="public_only",
            character_ids=characters,
            title=f"{format_name} invalid",
            logline="Invalid scene-role contract.",
            scenes=tuple(
                _scene(index, characters, role)
                for index, role in enumerate(roles, start=1)
            ),
        )


def test_public_preview_member_continuation_is_structurally_enforced() -> None:
    plan = ContentPackPlan(
        format="single_feature",
        editorial_archetype="public_preview_member_continuation",
        character_ids=("a",),
        title="Preview then continuation",
        logline="One coherent Pack with a public preview.",
        scenes=(
            _scene(1, ("a",), "preview", tier="public"),
            _scene(2, ("a",), "detail", tier="member", rating="adult"),
        ),
    )
    assert plan.scenes[0].publication_tier is PublicationTier.PUBLIC
    assert plan.scenes[1].publication_tier is PublicationTier.MEMBER
    assert plan.scenes[1].planned_content_rating is ContentRating.ADULT

    with pytest.raises(ValueError, match="preview scene"):
        ContentPackPlan(
            format="single_feature",
            editorial_archetype="public_preview_member_continuation",
            character_ids=("a",),
            title="Invalid preview",
            logline="First scene is not marked as preview.",
            scenes=(
                _scene(1, ("a",), "feature", tier="public"),
                _scene(2, ("a",), "detail", tier="member"),
            ),
        )


def test_public_scene_cannot_intentionally_plan_adult_content() -> None:
    with pytest.raises(ValueError, match="public scenes"):
        ContentPackPlan(
            format="single_feature",
            editorial_archetype="public_only",
            character_ids=("a",),
            title="Invalid public rating",
            logline="Adult content cannot be planned for public.",
            scenes=(
                _scene(
                    1,
                    ("a",),
                    "feature",
                    tier="public",
                    rating="adult",
                ),
            ),
        )


def _allow_profile() -> PolicyProfile:
    return PolicyProfile(
        id="allow",
        version="1",
        use_rules={
            "private": "allow",
            "public_free": "allow",
            "paid_membership": "allow",
            "commercial": "allow",
        },
        tier_rules={
            "public": "allow",
            "member": "allow",
            "private_review": "allow",
            "blocked": "block",
        },
    )


def _patreon_engine(
    tmp_path: Path,
) -> tuple[Database, CharacterRegistry, PolicyEngine]:
    database = Database(f"sqlite:///{(tmp_path / 'patreon.sqlite3').as_posix()}")
    database.migrate()
    characters = CharacterRegistry(database)
    characters.upsert(
        CharacterProfile(
            id="char-a",
            display_name="Character A",
            namespace="test",
            policy_profile="allow",
        )
    )
    registry = PolicyRegistry.with_packaged_defaults()
    registry.register(_allow_profile())
    engine = PolicyEngine(
        registry,
        characters,
        PolicyDecisionRepository(database),
        RightsConfig(
            default_profile="allow",
            platform_profile="patreon_2026_10",
        ),
    )
    return database, characters, engine


def test_patreon_policy_separates_access_tier_from_content_rating(
    tmp_path: Path,
) -> None:
    database, _, engine = _patreon_engine(tmp_path)

    public_adult = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="public-adult",
            character_ids=("char-a",),
            use_class=UseClass.PUBLIC_FREE,
            requested_tier=PublicationTier.PUBLIC,
            content_rating=ContentRating.ADULT,
        )
    )
    member_adult = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="member-adult",
            character_ids=("char-a",),
            use_class=UseClass.PAID_MEMBERSHIP,
            requested_tier=PublicationTier.MEMBER,
            content_rating=ContentRating.ADULT,
        )
    )
    public_suggestive = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="public-suggestive",
            character_ids=("char-a",),
            use_class=UseClass.PUBLIC_FREE,
            requested_tier=PublicationTier.PUBLIC,
            content_rating=ContentRating.SUGGESTIVE,
        )
    )
    prohibited = engine.evaluate(
        PolicyRequest(
            subject_type="scene",
            subject_id="prohibited",
            character_ids=("char-a",),
            use_class=UseClass.PAID_MEMBERSHIP,
            requested_tier=PublicationTier.MEMBER,
            content_rating=ContentRating.ADULT,
            content_labels=("sexualized_minor",),
        )
    )

    assert public_adult.outcome is PolicyOutcome.BLOCK
    assert public_adult.effective_tier is PublicationTier.BLOCKED
    assert member_adult.outcome is PolicyOutcome.ALLOW
    assert member_adult.effective_tier is PublicationTier.MEMBER
    assert public_suggestive.outcome is PolicyOutcome.REVIEW
    assert public_suggestive.effective_tier is PublicationTier.PRIVATE_REVIEW
    assert prohibited.outcome is PolicyOutcome.BLOCK
    assert "patreon_2026_10@2026-10-06" in public_adult.profile_versions

    packaged = PolicyRegistry.with_packaged_defaults().require("patreon_2026_10")
    assert packaged.checked_at is not None
    assert packaged.source_urls
    database.dispose()


def test_post_generation_policy_uses_planned_tier_not_temporary_review_tier(
    tmp_path: Path,
) -> None:
    database, _, engine = _patreon_engine(tmp_path)
    now = datetime.now(UTC)
    plan = _scene(1, ("char-a",), "feature", tier="public")

    with database.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state=PackState.EVALUATING.value,
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
                publication_tier=PublicationTier.PRIVATE_REVIEW.value,
                payload_json={"plan": plan.model_dump(mode="json")},
            )
        )

    service = PolicyApplicationService(database, engine)
    decision = service.evaluate_scene(
        "scene-1",
        use_class=UseClass.PUBLIC_FREE,
        content_rating=ContentRating.ADULT,
        phase="post_generation",
    )

    assert decision.requested_tier is PublicationTier.PUBLIC
    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.effective_tier is PublicationTier.BLOCKED
    database.dispose()


def test_archive_generates_patreon_post_package_with_public_member_mapping(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{(tmp_path / 'archive.sqlite3').as_posix()}")
    database.migrate()
    now = datetime.now(UTC)
    public_image = tmp_path / "public.png"
    member_image = tmp_path / "member.png"
    public_image.write_bytes(b"public-image")
    member_image.write_bytes(b"member-image")

    plan = ContentPackPlan(
        format="single_feature",
        editorial_archetype="public_preview_member_continuation",
        character_ids=("char-a",),
        title="Window Pack",
        logline="A coherent public preview with a member continuation.",
        scenes=(
            _scene(1, ("char-a",), "preview", tier="public"),
            _scene(2, ("char-a",), "detail", tier="member", rating="adult"),
        ),
    )

    with database.session() as session:
        session.add(
            PackRow(
                id="pack-1",
                state=PackState.EVALUATING.value,
                format_type=plan.format.value,
                payload_json={"plan": plan.model_dump(mode="json")},
                checkpoint_json={},
                created_at=now,
                updated_at=now,
            )
        )
        for ordinal, tier, image, rating in (
            (1, "public", public_image, "general"),
            (2, "member", member_image, "adult"),
        ):
            scene_id = f"scene-{ordinal}"
            attempt_id = f"attempt-{ordinal}"
            scene_plan = plan.scenes[ordinal - 1]
            session.add(
                SceneRow(
                    id=scene_id,
                    pack_id="pack-1",
                    ordinal=ordinal,
                    state=SceneState.ACCEPTED.value,
                    publication_tier=tier,
                    payload_json={
                        "plan": scene_plan.model_dump(mode="json"),
                        "content_rating": rating,
                        "content_labels": [],
                    },
                    selected_attempt_id=attempt_id,
                )
            )
            session.add(
                GenerationAttemptRow(
                    id=attempt_id,
                    scene_id=scene_id,
                    ordinal=1,
                    backend_status="completed",
                    seed=ordinal,
                    prompt="prompt",
                    negative_prompt="negative",
                    provenance_json={"output_paths": [str(image)]},
                    error_json=None,
                    created_at=now,
                )
            )

    result = PackArchive(
        database,
        tmp_path / "packs",
        PatreonConfig(enabled=True),
    ).finalize("pack-1")

    assert result.post_package_path is not None
    assert result.post_package_path.exists()
    package = json.loads(result.post_package_path.read_text(encoding="utf-8"))
    assert package["platform"] == "patreon"
    assert package["title"] == "Window Pack"
    assert package["editorial_archetype"] == "public_preview_member_continuation"
    assert package["public_scene_ids"] == ["scene-1"]
    assert package["member_scene_ids"] == ["scene-2"]
    assert package["review_scene_ids"] == []
    assert package["publication_ready"] is True
    assert package["scenes"][1]["content_rating"] == "adult"
    assert "char-a" in package["tags"]
    database.dispose()
