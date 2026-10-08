from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.config import load_settings
from artifex.llm.release_install import (
    extract_verified_zip,
    install_official_llama,
    official_release_assets,
)


TAG = "b12345"
NAME = "llama-b12345-bin-win-cuda-12.4-x64.zip"
DOWNLOAD = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{NAME}"
API = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}"


def _zip(*, member: str = "bin/llama-server.exe") -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, b"MZllama server fake")
        archive.writestr("bin/ggml-cuda.dll", b"MZfake-dll")
    return memory.getvalue()


def _client(
    content: bytes,
    *,
    digest: str | None = None,
    size: int | None = None,
    tag: str = TAG,
    source: str = DOWNLOAD,
    requests: list[str] | None = None,
) -> httpx.Client:
    expected = digest if digest is not None else hashlib.sha256(content).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(str(request.url))
        if str(request.url) == API:
            return httpx.Response(
                200,
                json={
                    "tag_name": tag,
                    "assets": [
                        {
                            "name": NAME,
                            "digest": f"sha256:{expected}",
                            "size": len(content) if size is None else size,
                            "browser_download_url": source,
                        }
                    ],
                },
            )
        if str(request.url) == DOWNLOAD:
            return httpx.Response(200, content=content)
        raise AssertionError(f"unexpected URL: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_approved_official_windows_asset_has_verified_digest() -> None:
    content = _zip()
    with _client(content) as client:
        assets = official_release_assets(TAG, client=client)
    assert len(assets) == 1
    assert assets[0].name == NAME
    assert assets[0].sha256 == hashlib.sha256(content).hexdigest()


def test_exact_download_sha_and_atomic_extract_with_dlls(
    tmp_path: Path,
) -> None:
    content = _zip()
    calls: list[str] = []
    with _client(content, requests=calls) as client:
        installed = install_official_llama(
            TAG, NAME, tmp_path / "llama bins", client=client
        )
        assert installed.installed
        assert installed.executable.read_bytes() == b"MZllama server fake"
        assert (installed.executable.parent / "ggml-cuda.dll").is_file()
        assert len(installed.executable_sha256) == 64
        reused = install_official_llama(
            TAG, NAME, tmp_path / "llama bins", client=client
        )
    assert not reused.installed
    assert reused.executable == installed.executable
    assert calls.count(DOWNLOAD) == 1
    assert not list((tmp_path / "llama bins").glob(".artifex-*"))


def test_rejects_digest_mismatch_and_cleans_staging(tmp_path: Path) -> None:
    content = _zip()
    folder = tmp_path / "bin"
    with _client(content, digest="0" * 64) as client, pytest.raises(ValueError, match="SHA-256/size"):
        install_official_llama(TAG, NAME, folder, client=client)
    assert not list(folder.iterdir())


def test_rejects_size_mismatch_without_install(tmp_path: Path) -> None:
    content = _zip()
    folder = tmp_path / "bin"
    with _client(content, size=len(content) + 20) as client, pytest.raises(ValueError, match="SHA-256/size"):
        install_official_llama(TAG, NAME, folder, client=client)
    assert not list(folder.iterdir())


@pytest.mark.parametrize(
    "member",
    [
        "../escape.txt",
        "/absolute.txt",
        "C:/system32/evil.exe",
        "bin/../other.exe",
        "bin\\..\\evil.exe",
    ],
)
def test_zip_slip_rejected_even_when_download_hash_matches(
    member: str, tmp_path: Path
) -> None:
    contents = _zip(member=member)
    with _client(contents) as client, pytest.raises(ValueError, match="Unsafe archive member"):
        install_official_llama(TAG, NAME, tmp_path / "bin", client=client)
    assert not (tmp_path / "escape.txt").exists()


def test_symlink_and_duplicate_archive_entries_are_rejected(tmp_path: Path) -> None:
    symlink_archive = tmp_path / "link.zip"
    with zipfile.ZipFile(symlink_archive, "w") as archive:
        link = zipfile.ZipInfo("llama-server.exe")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "../escape.exe")
    with pytest.raises(ValueError, match="symlink"):
        extract_verified_zip(symlink_archive, tmp_path / "extract")
    double = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(double, "w") as archive:
        archive.writestr("bin/llama-server.exe", b"MZ")
        archive.writestr("BIN/LLAMA-SERVER.EXE", b"fake")
    with pytest.raises(ValueError, match="case-colliding"):
        extract_verified_zip(double, tmp_path / "out")


def test_missing_executable_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "missing.zip"
    with zipfile.ZipFile(archive, "w") as zipout:
        zipout.writestr("README.md", "hello")
    with pytest.raises(ValueError, match="one llama-server.exe"):
        extract_verified_zip(archive, tmp_path / "output")


def test_missing_upstream_checksum_fails_closed() -> None:
    with _client(_zip(), digest="not-a-sha") as client:
        assert official_release_assets(TAG, client=client) == ()


def test_release_metadata_mismatch_or_wrong_url_fails_closed() -> None:
    with _client(_zip(), tag="b9999") as client, pytest.raises(ValueError, match="does not match"):
        official_release_assets(TAG, client=client)
    with _client(_zip(), source="https://evil.example.com/llama.zip") as client:
        assert official_release_assets(TAG, client=client) == ()


def test_path_and_tag_injection_rejected_before_http_or_disk(tmp_path: Path) -> None:
    with _client(_zip()) as client, pytest.raises(ValueError, match="tag"):
        official_release_assets("../main", client=client)
        with pytest.raises(ValueError, match="asset"):
            install_official_llama(TAG, "../../evil.zip", tmp_path, client=client)
    assert list(tmp_path.iterdir()) == []


def test_existing_unowned_install_is_not_overwritten(tmp_path: Path) -> None:
    existing = tmp_path / "llama-cpp-b12345-llama-b12345-bin-win-cuda-12.4-x64"
    existing.mkdir()
    (existing / "my-file.txt").write_text("keep", encoding="utf-8")
    with _client(_zip()) as client, pytest.raises(FileExistsError, match="Refusing to overwrite"):
        install_official_llama(TAG, NAME, tmp_path, client=client)
    assert (existing / "my-file.txt").read_text() == "keep"


def test_existing_install_with_tampered_binary_is_rejected(tmp_path: Path) -> None:
    with _client(_zip()) as client:
        installed = install_official_llama(TAG, NAME, tmp_path, client=client)
        installed.executable.write_text("tampered", encoding="utf-8")
        with pytest.raises(FileExistsError, match="cannot be trusted"):
            install_official_llama(TAG, NAME, tmp_path, client=client)


def test_receipt_does_not_allow_parent_directory_binaries(tmp_path: Path) -> None:
    with _client(_zip()) as client:
        installed = install_official_llama(TAG, NAME, tmp_path, client=client)
        receipt = installed.directory / ".artifex-install.json"
        data = json.loads(receipt.read_text(encoding="utf-8"))
        data["relative_executable"] = "../../llama-server.exe"
        receipt.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(FileExistsError, match="cannot be trusted"):
            install_official_llama(TAG, NAME, tmp_path, client=client)


def test_cli_assets_listing_and_config_update_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from artifex.llm.release_install import (
        LlamaInstallResult,
        OfficialLlamaAsset,
    )

    monkeypatch.setattr(
        "artifex.cli.official_release_assets",
        lambda tag: (
            OfficialLlamaAsset(
                tag=tag, name=NAME, sha256="a" * 64, bytes=100, url=DOWNLOAD
            ),
        ),
    )
    listed = CliRunner().invoke(app, ["onboard", "llama-assets", "--tag", TAG])
    assert listed.exit_code == 0, listed.output
    assert NAME in listed.output

    binary = tmp_path / "llama-server.exe"
    binary.touch()
    monkeypatch.setattr(
        "artifex.cli.install_official_llama",
        lambda tag, asset, output_dir: LlamaInstallResult(
            tag=tag, asset=asset, archive_sha256="a" * 64,
            executable=binary, executable_sha256="b" * 64,
            directory=tmp_path, installed=False,
        ),
    )
    config = tmp_path / "config.yaml"
    config.write_text("discord:\n  enabled: false\n", encoding="utf-8")
    nonconfigured = CliRunner().invoke(
        app, ["onboard", "llama-install", "--tag", TAG, "--asset", NAME]
    )
    assert nonconfigured.exit_code == 0, nonconfigured.output
    assert '"configured_path"' not in nonconfigured.output

    configured = CliRunner().invoke(
        app, [
            "onboard", "llama-install", "--tag", TAG, "--asset", NAME,
            "--config", str(config), "--configure",
        ],
    )
    assert configured.exit_code == 0, configured.output
    settings = load_settings(user_config=config, env={})
    assert settings.llm.server.enabled
    assert settings.llm.server.executable == str(binary)
    assert settings.discord.enabled is False


def test_configure_without_config_fails_before_download(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "artifex.cli.install_official_llama",
        lambda *args, **kwargs: pytest.fail("installer must not run"),
    )
    result = CliRunner().invoke(
        app, [
            "onboard", "llama-install", "--tag", TAG,
            "--asset", NAME, "--configure",
        ],
    )
    assert result.exit_code == 1
    assert "requires --config" in result.output
