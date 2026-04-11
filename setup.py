"""
TurboQuant vLLM Plugin -- setup.py

Registers the TurboQuantPlatform via the ``vllm.platform_plugins`` entry
point so that ``vllm serve <model> --attention-backend turboquant`` works.

Install in development mode:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="turboquant-vllm",
    version="0.1.0",
    description=(
        "TurboQuant KV cache compression plugin for vLLM -- "
        "3-bit asymmetric attention with near-zero accuracy loss"
    ),
    author="TurboQuant Contributors",
    license="MIT",
    python_requires=">=3.10",
    packages=["vllm_plugin"],
    py_modules=["turboquant", "lloyd_max", "compressors", "ollama_resolver"],
    install_requires=[
        "torch>=2.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.6.0", "gguf>=0.18.0"],
        "validate": [
            "transformers>=4.40.0",
            "accelerate>=0.25.0",
            "bitsandbytes>=0.43.0",
        ],
    },
    entry_points={
        "vllm.general_plugins": [
            "turboquant = vllm_plugin.platform:register_turboquant",
        ],
    },
)
