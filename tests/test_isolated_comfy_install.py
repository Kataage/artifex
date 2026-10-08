from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from artifex.cli import app
from artifex.comfy import isolated_install

SHA = "a" * 40
URL = "https://codeload.github.com/Comfy-Org/ComfyUI/zip/" + SHA


def _archive(*, path: str | None = None) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("ComfyUI-aaaa/main.py", "print('hello')")
        z.writestr("ComfyUI-aaaa/requirements.txt", "pillow>=11\n")
        if path is not None:
            z.writestr(path, "bad")
    return stream.getvalue()


def _client(payload: bytes, requests: list[str] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(str(request.url))
        assert str(request.url) == URL
        return httpx.Response(200, content=payload)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_isolated_install_pinned_source_without_running_code(tmp_path: Path) -> None:
    requests: list[str] = []
    archive = _archive()
    with _client(archive, requests) as client:
        result = isolated_install.install_isolated_comfy(SHA, tmp_path, client=client)
    assert requests == [URL]
    assert result.commit == SHA
    assert result.installed
    assert not result.dependencies_installed
    assert result.python_executable is None
    assert result.archive_sha256 == hashlib.sha256(archive).hexdigest()
    assert (result.comfy_directory / "main.py").is_file()
    assert (result.comfy_directory / "requirements.txt").is_file()
    receipt = json.loads(
        (result.directory / ".artifex-comfy-install.json").read_text(encoding="utf-8")
    )
    assert receipt["commit"] == SHA
    assert not list(tmp_path.glob(".artifex-comfy-*"))


def test_existing_target_not_overwritten_or_modified(tmp_path: Path) -> None:
    destination = tmp_path / ("comfyui-" + SHA[:12])
    destination.mkdir()
    important = destination / "operator.txt"
    important.write_text("untouched", encoding="utf-8")
    with _client(_archive()) as client, pytest.raises(FileExistsError, match="overwrite"):
        isolated_install.install_isolated_comfy(SHA, tmp_path, client=client)
    assert important.read_text(encoding="utf-8") == "untouched"


def test_wrong_checksum_fails_without_install(tmp_path: Path) -> None:
    with _client(_archive()) as client, pytest.raises(ValueError, match="SHA-256 mismatch"):
        isolated_install.install_isolated_comfy(
            SHA, tmp_path, client=client, expected_archive_sha256="0" * 64
        )
    assert list(tmp_path.iterdir()) == []


def test_matching_external_checksum_is_accepted(tmp_path: Path) -> None:
    payload = _archive()
    with _client(payload) as client:
        result = isolated_install.install_isolated_comfy(
            SHA, tmp_path, client=client,
            expected_archive_sha256=hashlib.sha256(payload).hexdigest()
        )
    assert result.installed


@pytest.mark.parametrize(
    "path",
    [
        "ComfyUI-aaaa/../../escape.py",
        "ComfyUI-aaaa/C:/windows/evil.py",
        "../outside.py",
        "ComfyUI-other/evil.py",
    ],
)
def test_archive_path_and_second_root_rejected(tmp_path: Path, path: str) -> None:
    archive = _archive(path=path)
    with _client(archive) as client, pytest.raises(ValueError):
        isolated_install.install_isolated_comfy(SHA, tmp_path, client=client)
    assert not (tmp_path / "escape.py").exists()
    assert not (tmp_path / ("comfyui-" + SHA[:12])).exists()


def test_zip_archive_symlink_is_rejected(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("ComfyUI-aaaa/main.py", "hi")
        z.writestr("ComfyUI-aaaa/requirements.txt", "")
        link = zipfile.ZipInfo("ComfyUI-aaaa/custom_nodes/link")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        z.writestr(link, "../../outside")
    with _client(stream.getvalue()) as client, pytest.raises(ValueError, match="file type"):
        isolated_install.install_isolated_comfy(SHA, tmp_path, client=client)


def test_archive_missing_required_comfy_files_is_rejected(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("ComfyUI-aaaa/main.py", "only one")
    with _client(stream.getvalue()) as client, pytest.raises(ValueError, match="valid ComfyUI"):
        isolated_install.install_isolated_comfy(SHA, tmp_path, client=client)
    assert not list(tmp_path.glob("comfyui-*"))


def test_requires_full_commit_and_explicit_gpu_backend_before_network(tmp_path: Path) -> None:
    with _client(_archive()) as client:
        with pytest.raises(ValueError, match="40-character"):
            isolated_install.install_isolated_comfy("master", tmp_path, client=client)
        with pytest.raises(ValueError, match="torch-backend"):
            isolated_install.install_isolated_comfy(
                SHA, tmp_path, install_dependencies=True, client=client
            )
        with pytest.raises(ValueError, match="require --install-deps"):
            isolated_install.install_isolated_comfy(
                SHA, tmp_path, torch_backend="cu130", client=client
            )
    assert not list(tmp_path.iterdir())


def test_isolated_uv_dependencies_use_final_path_and_do_not_touch_other_envs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_install.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        isolated_install.shutil, "which", lambda value: "C:/uv.exe" if value == "uv" else None
    )
    python = tmp_path / "host-python.exe"
    python.write_bytes(b"MZ")
    commands: list[tuple[list[str], Path]] = []

    def fake_uv(args: list[str], *, cwd: Path) -> None:
        commands.append((args, cwd))
        if args[1] == "venv":
            venv = Path(args[-1])
            target = venv / "Scripts" / "python.exe"
            target.parent.mkdir(parents=True)
            target.touch()

    monkeypatch.setattr(isolated_install, "_run_uv", fake_uv)
    with _client(_archive()) as client:
        result = isolated_install.install_isolated_comfy(
            SHA, tmp_path / "install",
            install_dependencies=True,
            torch_backend="cu130",
            python=python,
            client=client,
        )
    assert result.dependencies_installed
    assert result.python_executable is not None
    assert result.python_executable.is_file()
    assert len(commands) == 3
    assert commands[0][0][1] == "venv"
    assert "--index-url" in commands[1][0]
    assert "https://download.pytorch.org/whl/cu130" in commands[1][0]
    assert "-r" in commands[2][0]
    assert all(cwd == result.comfy_directory for _, cwd in commands)
    assert not list((tmp_path / "install").glob(".artifex-comfy-*"))


def test_failed_uv_dependency_install_removes_only_new_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_install.platform, "system", lambda: "Windows")
    monkeypatch.setattr(isolated_install.shutil, "which", lambda _: "C:/uv.exe")
    python = tmp_path / "python.exe"
    python.touch()
    unrelated = tmp_path / "existing-ComfyUI"
    unrelated.mkdir()
    (unrelated / "operator.txt").write_text("preserved", encoding="utf-8")

    def fail_uv(args: list[str], *, cwd: Path) -> None:
        raise RuntimeError("uv failed")

    monkeypatch.setattr(isolated_install, "_run_uv", fail_uv)
    with _client(_archive()) as client, pytest.raises(RuntimeError, match="uv failed"):
        isolated_install.install_isolated_comfy(
            SHA, tmp_path, install_dependencies=True,
            torch_backend="cpu", python=python, client=client,
        )
    assert (unrelated / "operator.txt").read_text(encoding="utf-8") == "preserved"
    assert not list(tmp_path.glob("comfyui-*"))


def test_cli_rejects_ambiguous_source_before_network() -> None:
    result = CliRunner().invoke(
        app, ["onboard", "comfy-install", "--commit", "latest"]
    )
    assert result.exit_code == 1
    assert "40-character" in result.output


def test_cli_passes_explicit_options_without_external_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def install(commit: str, output_root: Path, **kwargs: Any) -> isolated_install.ComfyInstallResult:
        seen.update({"commit": commit, "output_root": output_root, **kwargs})
        return isolated_install.ComfyInstallResult(
            commit=SHA, directory=tmp_path, comfy_directory=tmp_path / "ComfyUI",
            archive_sha256="0" * 64, dependencies_installed=False,
            python_executable=None, torch_backend=None, installed=True,
        )
    monkeypatch.setattr("artifex.cli.install_isolated_comfy", install)
    result = CliRunner().invoke(
        app, [
            "onboard", "comfy-install",
            "--commit", SHA, "--output-dir", str(tmp_path),
            "--archive-sha256", "1" * 64,
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["expected_archive_sha256"] == "1" * 64
    assert seen["install_dependencies"] is False
