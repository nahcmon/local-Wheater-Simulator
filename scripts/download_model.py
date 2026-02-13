from __future__ import annotations

from chronos import ChronosPipeline
import torch

MODEL_ID = "amazon/chronos-bolt-small"


def main() -> None:
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ChronosPipeline.from_pretrained(MODEL_ID, device_map=device_map, torch_dtype=torch_dtype)
    print(f"Model ready: {MODEL_ID}")


if __name__ == "__main__":
    main()
