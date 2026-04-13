"""Resolve Ollama model names to GGUF blobs and TurboQuant metadata."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_LAYER_MEDIA_TYPE = "application/vnd.ollama.image.model"
DEFAULT_OLLAMA_HOST = "registry.ollama.ai"
DEFAULT_OLLAMA_NAMESPACE = "library"


@dataclass(frozen=True)
class GGUFMetadata:
    architecture: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    context_length: int
    file_type: int | None


@dataclass(frozen=True)
class OllamaModel:
    name: str
    tag: str
    gguf_path: Path
    size_bytes: int
    metadata: GGUFMetadata | None = None


def resolve_model(name: str) -> OllamaModel:
    """Resolve an Ollama model reference to the local GGUF blob path."""
    parsed_name, tag, manifest_parts = _parse_model_ref(name)
    models_dir = _ollama_models_dir()
    manifest_path = models_dir / "manifests" / Path(*manifest_parts) / tag
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ollama manifest not found for {name!r}: {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_layer = _find_model_layer(manifest)
    digest = model_layer.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError(f"Invalid Ollama model digest in {manifest_path}: {digest!r}")

    gguf_path = models_dir / "blobs" / digest.replace(":", "-", 1)
    if not gguf_path.exists():
        raise FileNotFoundError(
            f"Ollama GGUF blob not found for {name!r}: {gguf_path}"
        )

    size_bytes = model_layer.get("size")
    if not isinstance(size_bytes, int):
        size_bytes = gguf_path.stat().st_size

    return OllamaModel(
        name=parsed_name,
        tag=tag,
        gguf_path=gguf_path,
        size_bytes=size_bytes,
    )


def read_gguf_metadata(path: str | os.PathLike[str]) -> GGUFMetadata:
    """Read TurboQuant-relevant architecture metadata from a GGUF file."""
    try:
        import gguf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'gguf' package is required to read GGUF metadata. "
            "Install it with: pip install gguf"
        ) from exc

    reader = gguf.GGUFReader(path)
    fields = reader.fields
    architecture = _as_string(_required_field(fields, "general.architecture"))

    num_heads = _as_int(_required_field(fields, f"{architecture}.attention.head_count"))
    num_kv_heads = _as_int(
        _optional_field(
            fields,
            f"{architecture}.attention.head_count_kv",
            default=num_heads,
        )
    )

    key_length = _optional_field(fields, f"{architecture}.attention.key_length")
    if key_length is None:
        embedding_length = _as_int(
            _required_field(fields, f"{architecture}.embedding_length")
        )
        if embedding_length % num_heads != 0:
            raise ValueError(
                f"{architecture}.embedding_length ({embedding_length}) must be "
                f"divisible by head_count ({num_heads})"
            )
        head_dim = embedding_length // num_heads
    else:
        head_dim = _as_int(key_length)

    return GGUFMetadata(
        architecture=architecture,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=_as_int(_required_field(fields, f"{architecture}.block_count")),
        context_length=_as_int(
            _required_field(fields, f"{architecture}.context_length")
        ),
        file_type=_as_optional_int(_optional_field(fields, "general.file_type")),
    )


def to_tq_env(metadata: GGUFMetadata) -> dict[str, str]:
    """Convert GGUF metadata into TurboQuant environment defaults."""
    return {
        "TQ_NUM_LAYERS": str(metadata.num_layers),
        "TQ_NUM_HEADS": str(metadata.num_heads),
        "TQ_NUM_KV_HEADS": str(metadata.num_kv_heads),
        "TQ_HEAD_DIM": str(metadata.head_dim),
        "TQ_MAX_SEQ_LEN": str(metadata.context_length),
    }


def tokenizer_for_model(model: OllamaModel, metadata: GGUFMetadata) -> str | None:
    """Return a conservative Hugging Face tokenizer hint for known Ollama GGUFs."""
    model_name = model.name.lower()
    architecture = metadata.architecture.lower()
    if "qwen2.5-coder" in model_name or architecture == "qwen2":
        return "Qwen/Qwen2.5-Coder-32B-Instruct"
    if architecture in {"qwen35moe", "qwen3moe"}:
        return "Qwen/Qwen3.5-35B-A3B"
    if architecture in {"qwen35", "qwen3"}:
        return "Qwen/Qwen3.5-27B"
    if architecture == "llama":
        return "meta-llama/Llama-3.3-70B-Instruct"
    return None


def _ollama_models_dir() -> Path:
    explicit_models = os.environ.get("OLLAMA_MODELS")
    if explicit_models:
        return Path(explicit_models).expanduser()

    harbor_cache = _harbor_ollama_cache()
    if harbor_cache:
        return harbor_cache / "models"

    return Path("~/.ollama/models").expanduser()


def _harbor_ollama_cache() -> Path | None:
    explicit_cache = os.environ.get("HARBOR_OLLAMA_CACHE")
    if explicit_cache:
        return Path(explicit_cache).expanduser()

    harbor_env = _harbor_home() / ".env"
    if not harbor_env.exists():
        return None

    values = _read_env_file(harbor_env)
    cache = values.get("HARBOR_OLLAMA_CACHE")
    if not cache:
        return None
    return Path(os.path.expandvars(cache)).expanduser()


def _harbor_home() -> Path:
    return Path(os.environ.get("HARBOR_HOME", "~/.harbor")).expanduser()


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            parsed = shlex.split(raw_value, comments=False, posix=True)
        except ValueError:
            parsed = [raw_value.strip().strip('"').strip("'")]
        values[key] = parsed[0] if parsed else ""
    return values


def _parse_model_ref(ref: str) -> tuple[str, str, list[str]]:
    if not ref:
        raise ValueError("Model name must not be empty")

    name_part = ref
    tag = "latest"
    last_component = ref.rsplit("/", 1)[-1]
    if ":" in last_component:
        name_part, tag = ref.rsplit(":", 1)
        if not tag:
            raise ValueError(f"Model tag must not be empty: {ref!r}")

    parts = [part for part in name_part.split("/") if part]
    if not parts:
        raise ValueError(f"Invalid Ollama model name: {ref!r}")

    if len(parts) == 1:
        manifest_parts = [DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_NAMESPACE, parts[0]]
        parsed_name = parts[0]
    elif len(parts) == 2:
        if _looks_like_host(parts[0]):
            manifest_parts = [parts[0], DEFAULT_OLLAMA_NAMESPACE, parts[1]]
            parsed_name = parts[1]
        else:
            manifest_parts = [DEFAULT_OLLAMA_HOST, parts[0], parts[1]]
            parsed_name = "/".join(parts)
    else:
        manifest_parts = parts
        parsed_name = (
            "/".join(parts[1:]) if _looks_like_host(parts[0]) else name_part
        )

    return parsed_name, tag, manifest_parts


def _looks_like_host(value: str) -> bool:
    return "." in value or ":" in value or value == "localhost"


def _find_model_layer(manifest: dict[str, Any]) -> dict[str, Any]:
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Ollama manifest is missing a layers array")

    for layer in layers:
        if (
            isinstance(layer, dict)
            and layer.get("mediaType") == MODEL_LAYER_MEDIA_TYPE
        ):
            return layer
    raise ValueError(
        f"Ollama manifest does not contain a {MODEL_LAYER_MEDIA_TYPE!r} layer"
    )


def _required_field(fields: dict[str, Any], key: str) -> Any:
    value = _optional_field(fields, key)
    if value is None:
        raise KeyError(f"GGUF metadata field {key!r} is required")
    return value


def _optional_field(
    fields: dict[str, Any],
    key: str,
    default: Any | None = None,
) -> Any:
    field = fields.get(key)
    if field is None:
        return default
    if hasattr(field, "contents"):
        return field.contents()
    return field


def _as_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8")
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return bytes(value).decode("utf-8")
    if hasattr(value, "tobytes"):
        return value.tobytes().decode("utf-8")
    return str(value)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected integer GGUF metadata value, got {value!r}"
        ) from exc


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _as_int(value)


def _model_to_json(model: OllamaModel, metadata: GGUFMetadata) -> dict[str, Any]:
    return {
        "name": model.name,
        "tag": model.tag,
        "gguf_path": str(model.gguf_path),
        "size_bytes": model.size_bytes,
        "metadata": asdict(metadata),
        "tq_env": to_tq_env(metadata),
        "tokenizer": tokenizer_for_model(model, metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve an Ollama model name to a GGUF path and TQ config."
    )
    parser.add_argument(
        "model",
        help="Ollama model reference, e.g. qwen2.5-coder:32b",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Only resolve the GGUF blob path; do not inspect GGUF metadata.",
    )
    args = parser.parse_args()

    model = resolve_model(args.model)
    if args.no_metadata:
        print(
            json.dumps(
                {
                    "name": model.name,
                    "tag": model.tag,
                    "gguf_path": str(model.gguf_path),
                    "size_bytes": model.size_bytes,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    metadata = read_gguf_metadata(model.gguf_path)
    print(json.dumps(_model_to_json(model, metadata), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
