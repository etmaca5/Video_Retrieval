"""
Download the VideoMAEv2-Base repository from Hugging Face.

Note: `transformers.AutoModel` requires a deep-learning backend (PyTorch/TF). If you're on
Python 3.14, PyTorch may not be available yet; this script still lets you download the
model files without installing torch.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Install with:\n"
            "  pip install huggingface_hub\n"
            "or:\n"
            "  conda install -c conda-forge huggingface_hub\n"
        ) from e

    out_dir = Path("model/VideoMAEv2-Base")
    out_dir.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id="OpenGVLab/VideoMAEv2-Base",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {local_path}")


if __name__ == "__main__":
    main()