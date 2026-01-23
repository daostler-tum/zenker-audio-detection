import argparse
import os
from typing import List

import numpy as np

from baselines.common import audio_features, data, utils


def _load_yamnet():
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_hub as hub  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Baseline A requires tensorflow and tensorflow_hub. "
            "Install them in your environment to use YAMNet."
        ) from exc

    model = hub.load("https://tfhub.dev/google/yamnet/1")
    return tf, model


def _embedding_for_file(
    *,
    fp: str,
    tf,
    model,
    cfg_audio: audio_features.AudioConfig,
    aggregate: List[str],
) -> np.ndarray:
    y, _ = audio_features.load_audio_mono(fp, cfg=cfg_audio)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)

    scores, embeddings, _ = model(waveform)
    emb = embeddings.numpy()  # [frames, 1024]

    parts: List[np.ndarray] = []
    if "mean" in aggregate:
        parts.append(np.mean(emb, axis=0))
    if "max" in aggregate:
        parts.append(np.max(emb, axis=0))
    if len(parts) == 0:
        parts.append(np.mean(emb, axis=0))

    return np.concatenate(parts, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--label_csv", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument(
        "--config", default=os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    ap.add_argument("--predefined_folds_dir", default=None)
    args = ap.parse_args()

    cfg = utils.load_yaml(args.config)
    task = str(cfg.get("task", "binary_stage2"))
    if task != "binary_stage2":
        raise ValueError(
            "yamnet_lr baseline currently supports only task=binary_stage2"
        )

    cfg_audio = audio_features.AudioConfig(
        sample_rate=int(cfg["sample_rate"]),
        clip_duration_s=float(cfg["clip_duration_s"]),
    )

    aggregate = list(cfg.get("embedding", {}).get("aggregate", ["mean", "max"]))

    tf, model = _load_yamnet()

    ds = data.index_dataset(
        data_root=args.data_root, label_csv=args.label_csv, task=task
    )

    cache_dir = utils.ensure_dir(os.path.join(args.output_dir, "embeddings_cache"))

    for fp in ds.filepaths:
        out_path = os.path.join(cache_dir, f"yamnet_{utils.stable_hash(fp)}.npy")
        if os.path.exists(out_path):
            continue
        v = _embedding_for_file(
            fp=fp, tf=tf, model=model, cfg_audio=cfg_audio, aggregate=aggregate
        )
        np.save(out_path, v)


if __name__ == "__main__":
    main()
