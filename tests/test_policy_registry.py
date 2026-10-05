from __future__ import annotations

from pathlib import Path

from artifex.policy import RightsPolicyRegistry


def test_packaged_default_policy_is_available() -> None:
    registry = RightsPolicyRegistry.with_packaged_defaults()

    default = registry.require("default")

    assert default.version == "1"
    assert default.allow_generation is True
    assert default.minimum_publication_tier.value == "public"


def test_policy_registry_loads_custom_yaml(tmp_path: Path) -> None:
    directory = tmp_path / "rights"
    directory.mkdir()
    (directory / "member.yaml").write_text(
        """
id: member_only
version: "2026-10-05"
allow_generation: true
minimum_publication_tier: member
requires_operator_review: false
allowed_formats:
  - single_feature
""",
        encoding="utf-8",
    )

    registry = RightsPolicyRegistry.with_packaged_defaults((directory,))

    profile = registry.require("member_only")
    assert profile.version == "2026-10-05"
    assert profile.minimum_publication_tier.value == "member"
    assert profile.allowed_formats[0].value == "single_feature"
