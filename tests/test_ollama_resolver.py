import json
import sys
from types import SimpleNamespace

import pytest


def test_resolve_model_reads_ollama_manifest(tmp_path, monkeypatch) -> None:
    from ollama_resolver import resolve_model

    store = tmp_path / "ollama"
    manifest_dir = (
        store / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5-coder"
    )
    manifest_dir.mkdir(parents=True)
    blob_dir = store / "blobs"
    blob_dir.mkdir()
    digest = "sha256:" + ("a" * 64)
    blob_path = blob_dir / ("sha256-" + ("a" * 64))
    blob_path.write_bytes(b"gguf")
    (manifest_dir / "32b").write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.projector",
                        "digest": "sha256:" + ("b" * 64),
                    },
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": digest,
                        "size": 4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_MODELS", str(store))

    model = resolve_model("qwen2.5-coder:32b")

    assert model.name == "qwen2.5-coder"
    assert model.tag == "32b"
    assert model.gguf_path == blob_path
    assert model.size_bytes == 4


def test_resolve_model_defaults_to_latest_tag(tmp_path, monkeypatch) -> None:
    from ollama_resolver import resolve_model

    store = tmp_path / "ollama"
    manifest_dir = (
        store / "manifests" / "registry.ollama.ai" / "library" / "llama3.3"
    )
    manifest_dir.mkdir(parents=True)
    blob_dir = store / "blobs"
    blob_dir.mkdir()
    digest = "sha256:" + ("c" * 64)
    blob_path = blob_dir / ("sha256-" + ("c" * 64))
    blob_path.write_bytes(b"gguf")
    (manifest_dir / "latest").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_MODELS", str(store))

    model = resolve_model("llama3.3")

    assert model.tag == "latest"
    assert model.gguf_path == blob_path


def test_resolve_model_reports_missing_manifest(tmp_path, monkeypatch) -> None:
    from ollama_resolver import resolve_model

    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "ollama"))

    with pytest.raises(FileNotFoundError, match="Ollama manifest not found"):
        resolve_model("missing:latest")


def test_resolve_model_uses_harbor_ollama_cache(tmp_path, monkeypatch) -> None:
    from ollama_resolver import resolve_model

    harbor_cache = tmp_path / "harbor-cache" / "ollama"
    store = harbor_cache / "models"
    manifest_dir = (
        store / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5-coder"
    )
    manifest_dir.mkdir(parents=True)
    blob_dir = store / "blobs"
    blob_dir.mkdir()
    digest = "sha256:" + ("d" * 64)
    blob_path = blob_dir / ("sha256-" + ("d" * 64))
    blob_path.write_bytes(b"gguf")
    (manifest_dir / "32b").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setenv("HARBOR_OLLAMA_CACHE", str(harbor_cache))

    model = resolve_model("qwen2.5-coder:32b")

    assert model.gguf_path == blob_path


def test_resolve_model_reads_harbor_env_file(tmp_path, monkeypatch) -> None:
    from ollama_resolver import resolve_model

    harbor_home = tmp_path / "harbor"
    harbor_cache = tmp_path / "cache with spaces" / "ollama"
    store = harbor_cache / "models"
    manifest_dir = (
        store / "manifests" / "registry.ollama.ai" / "library" / "llama3.3"
    )
    manifest_dir.mkdir(parents=True)
    blob_dir = store / "blobs"
    blob_dir.mkdir()
    digest = "sha256:" + ("e" * 64)
    blob_path = blob_dir / ("sha256-" + ("e" * 64))
    blob_path.write_bytes(b"gguf")
    (manifest_dir / "latest").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    harbor_home.mkdir()
    (harbor_home / ".env").write_text(
        f'HARBOR_OLLAMA_CACHE="{harbor_cache}"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.delenv("HARBOR_OLLAMA_CACHE", raising=False)
    monkeypatch.setenv("HARBOR_HOME", str(harbor_home))

    model = resolve_model("llama3.3")

    assert model.gguf_path == blob_path


def test_to_tq_env_maps_metadata_to_string_values() -> None:
    from ollama_resolver import GGUFMetadata, to_tq_env

    metadata = GGUFMetadata(
        architecture="qwen2",
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        num_layers=64,
        context_length=32768,
        file_type=15,
    )

    assert to_tq_env(metadata) == {
        "TQ_NUM_LAYERS": "64",
        "TQ_NUM_HEADS": "64",
        "TQ_NUM_KV_HEADS": "8",
        "TQ_HEAD_DIM": "128",
        "TQ_MAX_SEQ_LEN": "32768",
    }


def test_read_gguf_metadata_uses_key_length(monkeypatch, tmp_path) -> None:
    from ollama_resolver import read_gguf_metadata

    reader = SimpleNamespace(
        fields={
            "general.architecture": SimpleNamespace(contents=lambda: "qwen2"),
            "qwen2.attention.head_count": SimpleNamespace(contents=lambda: 64),
            "qwen2.attention.head_count_kv": SimpleNamespace(contents=lambda: 8),
            "qwen2.attention.key_length": SimpleNamespace(contents=lambda: 128),
            "qwen2.block_count": SimpleNamespace(contents=lambda: 64),
            "qwen2.context_length": SimpleNamespace(contents=lambda: 32768),
            "general.file_type": SimpleNamespace(contents=lambda: 15),
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "gguf",
        SimpleNamespace(GGUFReader=lambda path: reader),
    )

    metadata = read_gguf_metadata(str(tmp_path / "model.gguf"))

    assert metadata.architecture == "qwen2"
    assert metadata.num_heads == 64
    assert metadata.num_kv_heads == 8
    assert metadata.head_dim == 128
    assert metadata.num_layers == 64
    assert metadata.context_length == 32768
    assert metadata.file_type == 15


def test_read_gguf_metadata_falls_back_to_embedding_length(monkeypatch, tmp_path) -> None:
    from ollama_resolver import read_gguf_metadata

    reader = SimpleNamespace(
        fields={
            "general.architecture": SimpleNamespace(contents=lambda: b"llama"),
            "llama.attention.head_count": SimpleNamespace(contents=lambda: 32),
            "llama.attention.head_count_kv": SimpleNamespace(contents=lambda: 8),
            "llama.embedding_length": SimpleNamespace(contents=lambda: 4096),
            "llama.block_count": SimpleNamespace(contents=lambda: 80),
            "llama.context_length": SimpleNamespace(contents=lambda: 131072),
            "general.file_type": SimpleNamespace(contents=lambda: 10),
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "gguf",
        SimpleNamespace(GGUFReader=lambda path: reader),
    )

    metadata = read_gguf_metadata(str(tmp_path / "model.gguf"))

    assert metadata.architecture == "llama"
    assert metadata.head_dim == 128

