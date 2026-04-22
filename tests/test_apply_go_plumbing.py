from __future__ import annotations

import subprocess
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_apply_go_plumbing_patches_kv_estimator_and_logs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "patches" / "stage2-qjl" / "apply_go_plumbing.sh"
    ollama_dir = tmp_path / "ollama"

    _write(
        ollama_dir / "ml" / "backend.go",
        """package ml

const (
\tDTypeInvalid = iota
\tDTypeMXFP4
)
""",
    )
    _write(
        ollama_dir / "ml" / "backend" / "ggml" / "ggml.go",
        """package ggml

import "C"

func (t *Tensor) DType() ml.DType {
\tswitch t.t._type {
\tcase C.GGML_TYPE_MXFP4:
\t\treturn ml.DTypeMXFP4
\tdefault:
\t\treturn ml.DTypeInvalid
\t}
}

func ggmlDType(dtype ml.DType) C.enum_ggml_type {
\tswitch dtype {
\tcase ml.DTypeMXFP4:
\t\treturn C.GGML_TYPE_MXFP4
\tdefault:
\t\treturn C.GGML_TYPE_F16
\t}
}

func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
\treturn &Tensor{
\t\tb: t.b,
\t\tt: C.ggml_cpy(ctx.(*Context).ctx, t.t, t2.(*Tensor).t),
\t}
}
""",
    )
    _write(
        ollama_dir / "runner" / "ollamarunner" / "cache.go",
        """package ollamarunner

import (
\t"log/slog"
)

func kvCacheTypeFromStr(s string) ml.DType {
\tswitch s {
\tcase "q4_0":
\t\treturn ml.DTypeQ40
\tdefault:
\t\treturn ml.DTypeF16
\t}
}

func NewInputCache(model model.Model, kvCacheType string, kvSize int32, numSlots int, batchSize int, multiUserCache bool) (*InputCache, error) {
\tcache := model.Config().Cache
\tif cache != nil {
\t\tcache.Init(model.Backend(), kvCacheTypeFromStr(kvCacheType), numSlots, int(kvSize), batchSize)
\t}
\tslog.Debug("cache ready", "type", kvCacheType)
\treturn nil, nil
}
""",
    )
    _write(
        ollama_dir / "llama" / "llama.go",
        """package llama

func kvCacheTypeFromStr(s string) C.enum_ggml_type {
\tswitch s {
\tcase "q4_0":
\t\treturn C.GGML_TYPE_Q4_0
\tdefault:
\t\treturn C.GGML_TYPE_F16
\t}
}
""",
    )
    _write(
        ollama_dir / "fs" / "ggml" / "ggml.go",
        """package ggml

import (
\t"fmt"
)

func kvCacheBytesPerElement(cacheType string) float64 {
\tswitch cacheType {
\tcase "q8_0":
\t\treturn 1
\tdefault:
\t\treturn 2
\t}
}

func GraphSize(kvCacheType string) int {
\tbpe := kvCacheBytesPerElement(kvCacheType)
\tfmt.Println(bpe)
\treturn int(bpe)
}
""",
    )

    subprocess.run(["bash", str(script), str(ollama_dir)], check=True, text=True)
    subprocess.run(["bash", str(script), str(ollama_dir)], check=True, text=True)

    backend_go = (ollama_dir / "ml" / "backend.go").read_text(encoding="utf-8")
    cache_go = (ollama_dir / "runner" / "ollamarunner" / "cache.go").read_text(
        encoding="utf-8"
    )
    fs_ggml_go = (ollama_dir / "fs" / "ggml" / "ggml.go").read_text(encoding="utf-8")

    assert backend_go.count("DTypeTQ4P_D128") == 1
    assert 'case "tq4p_d128":\n\t\treturn 69.0 / 128.0' in fs_ggml_go
    assert 'case "tq4p_d256":\n\t\treturn 133.0 / 256.0' in fs_ggml_go
    assert fs_ggml_go.count("TQ4P: KV cache memory estimate") == 1
    assert cache_go.count("TQ4P: KV cache type configured") == 1
    assert '"log/slog"' in cache_go
    assert '"log/slog"' in fs_ggml_go
