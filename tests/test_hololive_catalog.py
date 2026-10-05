from __future__ import annotations

from pathlib import Path

from artifex.characters import CharacterRegistry, HololiveCatalog
from artifex.db import Database
from artifex.domain import CharacterStatus


def test_packaged_hololive_catalog_has_complete_stable_scope() -> None:
    catalog = HololiveCatalog.packaged()
    profiles = catalog.profiles
    by_id = {profile.id: profile for profile in profiles}

    assert catalog.catalog_id == "hololive"
    assert catalog.catalog_version == "2026.10.06.1"
    assert len(profiles) == 88
    assert len(by_id) == 88
    assert {profile.branch for profile in profiles} == {
        "jp",
        "id",
        "en",
        "dev_is",
        "asobi",
        "cn",
    }
    assert sum(profile.status is CharacterStatus.ACTIVE for profile in profiles) == 67
    assert sum(profile.status is CharacterStatus.GRADUATED for profile in profiles) == 16
    assert sum(profile.status is CharacterStatus.TERMINATED for profile in profiles) == 3
    assert sum(profile.status is CharacterStatus.AFFILIATE for profile in profiles) == 2

    assert by_id["watson_amelia"].status is CharacterStatus.AFFILIATE
    assert by_id["sakamata_chloe"].status is CharacterStatus.AFFILIATE
    assert by_id["amane_kanata"].status is CharacterStatus.GRADUATED
    assert by_id["uruha_rushia"].status is CharacterStatus.TERMINATED
    assert by_id["hitomi_chris"].status is CharacterStatus.TERMINATED
    assert by_id["artia"].branch == "cn"
    assert by_id["achichi_mela"].branch == "asobi"

    assert all(profile.namespace == "hololive" for profile in profiles)
    assert all("holostars" not in (profile.group or "").casefold() for profile in profiles)
    assert all(profile.canonical_tags for profile in profiles)
    assert all(profile.policy_profile for profile in profiles)
    assert all(profile.provenance for profile in profiles)
    assert all(profile.reference_image_dirs for profile in profiles)
    assert all(profile.outfits for profile in profiles)


def test_bootstrap_materializes_loadable_catalog_and_audits_complete(
    tmp_path: Path,
) -> None:
    catalog = HololiveCatalog.packaged()
    profile_dir = tmp_path / "profiles" / "characters"
    target = profile_dir / "hololive.yaml"

    assert catalog.materialize(target) == target
    assert target.exists()

    database = Database(f"sqlite:///{(tmp_path / 'catalog.sqlite3').as_posix()}")
    database.migrate()
    registry = CharacterRegistry(database)
    try:
        assert registry.load_directories((profile_dir,)) == 88
        report = catalog.audit(
            registry,
            (profile_dir,),
            minimum_readiness=0.5,
        )

        assert report.loaded_count == 88
        assert report.expected_count == 88
        assert report.status_counts == {
            "active": 67,
            "affiliate": 2,
            "graduated": 16,
            "terminated": 3,
        }
        assert report.catalog_complete
        assert report.production_ready
        assert not report.missing_ids
        assert not report.unexpected_ids
        assert not report.duplicate_profile_ids
        assert not report.excluded_namespace_ids
    finally:
        database.dispose()


def test_catalog_audit_detects_duplicate_and_excluded_namespace(
    tmp_path: Path,
) -> None:
    catalog = HololiveCatalog.packaged()
    profile_dir = tmp_path / "profiles" / "characters"
    catalog.materialize(profile_dir / "hololive.yaml")

    (profile_dir / "duplicate.yaml").write_text(
        """
id: tokino_sora
display_name: duplicate
namespace: hololive
canonical_tags: [tokino_sora]
policy_profile: unconfigured
readiness: 0.5
""",
        encoding="utf-8",
    )
    (profile_dir / "holostars.yaml").write_text(
        """
id: excluded_star
display_name: Excluded Star
namespace: holostars
branch: holostars
canonical_tags: [excluded_star]
policy_profile: unconfigured
readiness: 0.5
""",
        encoding="utf-8",
    )

    database = Database(f"sqlite:///{(tmp_path / 'audit.sqlite3').as_posix()}")
    database.migrate()
    registry = CharacterRegistry(database)
    try:
        registry.load_directories((profile_dir,))
        report = catalog.audit(
            registry,
            (profile_dir,),
            minimum_readiness=0.5,
        )

        assert "tokino_sora" in report.duplicate_profile_ids
        assert "excluded_star" in report.excluded_namespace_ids
        assert not report.catalog_complete
        assert not report.production_ready
    finally:
        database.dispose()


def test_catalog_materialize_requires_explicit_force(tmp_path: Path) -> None:
    catalog = HololiveCatalog.packaged()
    target = tmp_path / "hololive.yaml"
    catalog.materialize(target)

    try:
        catalog.materialize(target)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError without overwrite")

    catalog.materialize(target, overwrite=True)
