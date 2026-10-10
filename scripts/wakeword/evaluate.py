"""Evaluate the "Hey DrivePilot" model by streaming audio through the exact runtime API.

Run with the MAIN venv (the one the app uses), from the repo root:

    .venv/Scripts/python scripts/wakeword/evaluate.py
    .venv/Scripts/python scripts/wakeword/evaluate.py --quick      # small subsets, for smoke tests

Every audio condition is streamed through

    openwakeword.Model(wakeword_models=["models/wakeword/hey_drivepilot.onnx"], inference_framework="onnx")

in 1280-sample (80 ms) int16 chunks with ``model.predict(chunk)``; the model is reset between streams.
Raw per-frame scores are recorded and a detection rule is applied afterwards:

    a frame *fires* if its score >= threshold (and, if patience k > 1, the k-1 previous raw scores too),
    and no frame fired in the preceding ``debounce`` seconds,

which is what ``Model.predict(..., threshold=..., debounce_time=...)`` does. (openwakeword 0.6.0's own
``patience`` option is not used: it compares against already-zeroed buffer values, so it can never fire
— see the README.)

Data (none of it used for training):
  * positives: Piper clips from held-out test voices/speakers (scripts/wakeword/common.py), streamed
    with 2 s lead-in and 1.5 s tail; a detection counts if it fires between the clip start and 1 s
    after its end. Conditions: clean (zero padding, like Model.predict_clip), held-out real car noise
    (DEMAND TCAR, a car/recording never seen in training) at 10 dB and 5 dB SNR (speech RMS vs noise
    RMS), 10 dB + held-out room impulse responses, and wake phrase directly followed by a command.
  * false accepts: openWakeWord's validation_set_features.npy (~11 h DiPCo / Santa Barbara corpus /
    MUSDB features, fed to the same ONNX session; it was also used for checkpoint selection and
    threshold choice), LibriSpeech test-clean + test-other (~10.5 h real read speech, streamed),
    LibriSpeech test-clean with DEMAND car noise at 10 dB, DEMAND TCAR noise alone, and the held-out
    adversarial near-miss phrases (held-out speakers *and* held-out texts).

The threshold (and patience) is chosen on *validation* data only — validation speakers' recall in car
noise and false accepts on validation_set_features.npy — then all test numbers are reported at it.
Results go to models/wakeword/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
from common import (
    CHUNK_SAMPLES,
    CLIPS_DIR,
    DATA_DIR,
    LIBRISPEECH_DIR,
    MODEL_NAME,
    MODELS_DIR,
    POSITIVE_EXCLUDED_VOICES,
    REPO_ROOT,
    SAMPLE_RATE,
    VALIDATION_FEATURES,
    setup_logging,
    stable_seed,
    write_json,
)

log = logging.getLogger("evaluate")

LEAD_S = 2.0
TAIL_S = 1.5
DETECT_AFTER_END_S = 1.0
SPEECH_RMS_DBFS = (-32.0, -20.0)  # random per-stream speech level
DEBOUNCE_S = 2.0
THRESHOLDS = [round(x, 3) for x in np.concatenate([np.arange(0.05, 0.95, 0.05), [0.95, 0.97, 0.98, 0.99]])]
PATIENCES = [1, 2, 3]
FA_SELECTION_LIMIT = 0.25  # max false accepts / hour on validation_set_features when choosing a threshold
# Among thresholds within this much of the best validation recall, the highest threshold is chosen
# (keeps a margin above the score level where validation false accepts start to appear).
RECALL_TOLERANCE = 0.01
TARGETS = {"recall_clean": 0.95, "recall_car_10db": 0.90, "false_accepts_per_hour": 0.5}


# ---------------------------------------------------------------------------------------------
# Detection rule
# ---------------------------------------------------------------------------------------------
def fired_frames(scores: np.ndarray, threshold: float, patience: int = 1, debounce_s: float = DEBOUNCE_S) -> np.ndarray:
    """Indices of frames that fire under threshold + patience + debounce (see module docstring)."""
    above = scores >= threshold
    if patience > 1:
        run = np.convolve(above.astype(np.int32), np.ones(patience, dtype=np.int32), mode="full")[: len(above)]
        above = run >= patience
    cand = np.flatnonzero(above)
    if len(cand) == 0:
        return cand
    gap = math.ceil(debounce_s / (CHUNK_SAMPLES / SAMPLE_RATE))
    out, last = [], -(10**9)
    for i in cand:
        if i - last > gap:
            out.append(i)
            last = i
    return np.array(out, dtype=np.int64)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


# ---------------------------------------------------------------------------------------------
# Stream construction (runs in worker processes)
# ---------------------------------------------------------------------------------------------
_MODEL = None
_POOLS: dict = {}
_RIRS = None


def _init_worker(model_path: str) -> None:
    global _MODEL
    import openwakeword

    _MODEL = openwakeword.Model(wakeword_models=[model_path], inference_framework="onnx")


def _test_car_pool():
    from augment_features import NoisePool

    if "test_car" not in _POOLS:
        _POOLS["test_car"] = NoisePool("test_car")
    return _POOLS["test_car"]


def _test_rirs():
    global _RIRS
    if _RIRS is None:
        from augment_features import load_rirs

        _RIRS = load_rirs("test")
    return _RIRS


def _load(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: {sr} Hz")
    return x


def _stream_scores(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Reset the model and stream ``audio`` (float, [-1, 1]) in 80 ms int16 chunks; return scores."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    pad = (-len(pcm)) % CHUNK_SAMPLES
    pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])
    _MODEL.reset()
    scores = np.empty(len(pcm) // CHUNK_SAMPLES, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(len(scores)):
        scores[i] = _MODEL.predict(pcm[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES])[MODEL_NAME]
    return scores, (time.perf_counter() - t0) / max(1, len(scores))


def _utterance_task(task: dict) -> dict:
    """Build lead-in + utterance + tail for one clip and condition, stream it, return scores."""
    from augment_features import apply_rir, mix_at_snr, rms

    rng = np.random.default_rng(stable_seed(task["condition"], task["path"]))
    speech = _load(CLIPS_DIR / task["path"])
    if task.get("rir"):
        rirs = _test_rirs()
        speech = apply_rir(speech, rirs[rng.integers(len(rirs))])
    level_db = rng.uniform(*SPEECH_RMS_DBFS)
    speech = speech * (10 ** (level_db / 20) / rms(speech))
    lead, tail = int(LEAD_S * SAMPLE_RATE), int(TAIL_S * SAMPLE_RATE)
    audio = np.zeros(lead + len(speech) + tail, dtype=np.float32)
    span = (lead, lead + len(speech))
    audio[span[0] : span[1]] = speech
    if task.get("snr") is not None:
        noise = _test_car_pool().sample(len(audio), rng)
        audio = mix_at_snr(audio, noise, task["snr"], span)
    scores, t = _stream_scores(audio)
    return {"key": task["key"], "scores": scores, "span": span, "n_samples": len(audio), "sec_per_chunk": t}


def _long_task(task: dict) -> dict:
    """Stream a long recording (LibriSpeech chapter, optionally + car noise; or car noise alone)."""
    from augment_features import mix_at_snr

    rng = np.random.default_rng(stable_seed(task["key"]))
    if task["kind"] == "librispeech":
        audio = np.concatenate([_load(Path(p)) for p in task["files"]])
        if task.get("snr") is not None:
            noise = _test_car_pool().sample(len(audio), rng)
            audio = mix_at_snr(audio, noise, task["snr"], (0, len(audio)))
    else:  # car noise only: one 30 s DEMAND TCAR segment
        pool = _test_car_pool()
        k = task["segment"]
        audio = np.asarray(pool.data[pool.starts[k] : pool.starts[k] + pool.lengths[k]]).astype(np.float32) / 32768
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    scores, t = _stream_scores(audio)
    return {"key": task["key"], "scores": scores, "n_samples": len(audio), "sec_per_chunk": t}


# ---------------------------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------------------------
def read_manifest(set_name: str) -> list[dict]:
    """Manifest rows whose WAV exists, deduplicated and sorted."""
    rows = {}
    for line in (CLIPS_DIR / set_name / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "positive" in set_name and row["voice"] in POSITIVE_EXCLUDED_VOICES:
            continue  # see common.POSITIVE_EXCLUDED_VOICES
        if (CLIPS_DIR / row["path"]).exists():
            rows[row["path"]] = row
    return [rows[k] for k in sorted(rows)]


def run_tasks(fn, tasks: list[dict], model_path: str, workers: int, label: str) -> dict[str, dict]:
    """Run stream tasks in a process pool; return results keyed by task key."""
    t0 = time.time()
    out = {}
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model_path,)) as pool:
        for res in pool.imap_unordered(fn, tasks, chunksize=4):
            out[res["key"]] = res
    log.info("%s: %d streams in %.1f min", label, len(tasks), (time.time() - t0) / 60)
    return out


def utterance_detected(res: dict, threshold: float, patience: int) -> tuple[bool, int]:
    """(detected within [clip start, clip end + 1 s], number of fires before the clip start)."""
    fires = fired_frames(res["scores"], threshold, patience)
    frame_end = (fires + 1) * CHUNK_SAMPLES
    start, end = res["span"]
    hit = bool(np.any((frame_end >= start) & (frame_end <= end + DETECT_AFTER_END_S * SAMPLE_RATE)))
    early = int(np.sum(frame_end < start))
    return hit, early


def validation_feature_scores(model_path: str) -> np.ndarray:
    """Scores of the ONNX model on every 16-frame window of validation_set_features.npy.

    Uses the session inside an ``openwakeword.Model`` (same model, same runtime); the features are
    precomputed embeddings, so the melspectrogram/embedding streaming stage is skipped here.
    """
    import openwakeword

    model = openwakeword.Model(wakeword_models=[model_path], inference_framework="onnx")
    sess = model.models[MODEL_NAME]
    n_frames = model.model_inputs[MODEL_NAME]
    feats = np.load(VALIDATION_FEATURES).astype(np.float32)
    name = sess.get_inputs()[0].name
    scores = np.empty(len(feats) - n_frames + 1, dtype=np.float32)
    for i in range(len(scores)):
        scores[i] = sess.run(None, {name: feats[None, i : i + n_frames]})[0][0, 0]
    return scores


def summarize_recall(results: dict, keys: list[str], threshold: float, patience: int) -> dict:
    """Recall + Wilson CI + early-fire count for a set of utterance streams."""
    hits, early = 0, 0
    for k in keys:
        h, e = utterance_detected(results[k], threshold, patience)
        hits += h
        early += e
    return {
        "n": len(keys),
        "detected": hits,
        "recall": round(hits / max(1, len(keys)), 4),
        "recall_95ci": wilson(hits, len(keys)),
        "fires_before_utterance": early,
    }


def summarize_fa(results: dict, keys: list[str], threshold: float, patience: int) -> dict:
    """False accepts and hours over a set of negative streams."""
    n_fa = sum(len(fired_frames(results[k]["scores"], threshold, patience)) for k in keys)
    hours = sum(results[k]["n_samples"] for k in keys) / SAMPLE_RATE / 3600
    return {"hours": round(hours, 3), "false_accepts": n_fa, "per_hour": round(n_fa / hours, 3) if hours else None}


def file_sha256(path: str) -> str:
    """SHA-256 of a file (ties cached scores to the exact model)."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_env() -> dict:
    """Interpreter and package versions used for this evaluation."""
    import sys
    from importlib.metadata import version

    return {"python": sys.executable, **{pkg: version(pkg) for pkg in ("openwakeword", "onnxruntime", "numpy")}}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(MODELS_DIR / f"{MODEL_NAME}.onnx"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--quick", action="store_true", help="small subsets for a smoke test")
    parser.add_argument("--output", default=str(MODELS_DIR / "metrics.json"))
    parser.add_argument("--scores-cache", default=str(DATA_DIR / "eval_scores.pkl"),
                        help="raw per-frame scores are saved here after streaming")
    parser.add_argument("--from-scores", action="store_true",
                        help="recompute metrics from --scores-cache instead of streaming again")
    args = parser.parse_args()
    setup_logging("evaluate")
    model_path = str(Path(args.model).resolve())
    limit = 60 if args.quick else None

    # ---- streams -----------------------------------------------------------------------------
    test_pos = read_manifest("test/positive")[:limit]
    test_cmd = read_manifest("test/positive_command")[:limit]
    test_neg = read_manifest("test/negative")[:limit]
    val_pos = read_manifest("val/positive")[:limit]
    conditions = {
        "clean": {"snr": None},
        "car_10db": {"snr": 10.0},
        "car_5db": {"snr": 5.0},
        "car_10db_reverb": {"snr": 10.0, "rir": True},
    }
    utt_tasks = []
    groups: dict[str, list[str]] = {}

    def add(group: str, rows: list[dict], cond: dict) -> None:
        for r in rows:
            key = f"{group}|{r['path']}"
            utt_tasks.append({"key": key, "path": r["path"], "condition": group, **cond})
            groups.setdefault(group, []).append(key)

    for name, cond in conditions.items():
        add(f"test_pos/{name}", test_pos, cond)
    add("test_cmd/clean", test_cmd, {"snr": None})
    add("test_cmd/car_10db", test_cmd, {"snr": 10.0})
    add("test_adv/clean", test_neg, {"snr": None})
    add("test_adv/car_10db", test_neg, {"snr": 10.0})
    add("val_pos/car_10db", val_pos, {"snr": 10.0})
    add("val_pos/clean", val_pos, {"snr": None})

    chapters: dict[str, list[str]] = {}
    for f in sorted(LIBRISPEECH_DIR.rglob("*.flac")):
        chapters.setdefault(f"{f.parent.parent.parent.name}/{f.parent.parent.name}/{f.parent.name}", []).append(str(f))
    chapter_items = sorted(chapters.items())[: (6 if args.quick else None)]
    long_tasks = [{"key": f"libri/{c}", "kind": "librispeech", "files": fs} for c, fs in chapter_items]
    long_tasks += [
        {"key": f"libri_car10/{c}", "kind": "librispeech", "files": fs, "snr": 10.0}
        for c, fs in chapter_items
        if c.startswith("test-clean")
    ]
    from augment_features import NoisePool

    n_car_segments = len(NoisePool("test_car").starts)
    long_tasks += [
        {"key": f"carnoise/{k}", "kind": "carnoise", "segment": k}
        for k in range(n_car_segments if not args.quick else 4)
    ]

    log.info("model: %s", model_path)
    log.info("%d utterance streams, %d long streams", len(utt_tasks), len(long_tasks))
    cache = Path(args.scores_cache)
    if args.from_scores:
        with open(cache, "rb") as f:
            cached = pickle.load(f)
        if cached["model_sha256"] != file_sha256(model_path):
            raise RuntimeError(f"{cache} was computed for a different model")
        utt, longs, val_feat_scores = cached["utt"], cached["longs"], cached["val_feat_scores"]
        missing = {t["key"] for t in utt_tasks + long_tasks} - set(utt) - set(longs)
        if missing:
            raise RuntimeError(f"{len(missing)} streams missing from {cache}")
        log.info("loaded raw scores from %s", cache)
    else:
        utt = run_tasks(_utterance_task, utt_tasks, model_path, args.workers, "utterance streams")
        longs = run_tasks(_long_task, long_tasks, model_path, args.workers, "long streams")
        t0 = time.time()
        val_feat_scores = validation_feature_scores(model_path)
        log.info("validation_set_features: %d windows scored in %.1f s", len(val_feat_scores), time.time() - t0)
        if not args.quick:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump({"model_sha256": file_sha256(model_path), "utt": utt, "longs": longs,
                             "val_feat_scores": val_feat_scores}, f)

    libri_keys = [k for k in longs if k.startswith("libri/")]
    libri_car_keys = [k for k in longs if k.startswith("libri_car10/")]
    car_keys = [k for k in longs if k.startswith("carnoise/")]
    val_feat = {"val_features": {"scores": val_feat_scores, "n_samples": len(val_feat_scores) * CHUNK_SAMPLES}}

    # ---- threshold selection on validation data ------------------------------------------------
    sweep = []
    for patience in PATIENCES:
        for thr in THRESHOLDS:
            row = {"threshold": thr, "patience": patience}
            row["val_recall_car_10db"] = summarize_recall(utt, groups["val_pos/car_10db"], thr, patience)["recall"]
            row["val_recall_clean"] = summarize_recall(utt, groups["val_pos/clean"], thr, patience)["recall"]
            row["val_features_fa_per_hour"] = summarize_fa(val_feat, ["val_features"], thr, patience)["per_hour"]
            for name in conditions:
                row[f"test_recall_{name}"] = summarize_recall(utt, groups[f"test_pos/{name}"], thr, patience)["recall"]
            row["test_recall_command_clean"] = summarize_recall(utt, groups["test_cmd/clean"], thr, patience)["recall"]
            row["librispeech_fa_per_hour"] = summarize_fa(longs, libri_keys, thr, patience)["per_hour"]
            row["librispeech_car10_fa_per_hour"] = summarize_fa(longs, libri_car_keys, thr, patience)["per_hour"]
            row["adversarial_clean_triggered_frac"] = round(
                float(
                    np.mean([len(fired_frames(utt[k]["scores"], thr, patience)) > 0 for k in groups["test_adv/clean"]])
                ),
                4,
            )
            sweep.append(row)
    eligible = [r for r in sweep if r["val_features_fa_per_hour"] <= FA_SELECTION_LIMIT]
    if not eligible:
        log.warning("no threshold meets the validation false-accept limit; using the strictest")
        eligible = [max(sweep, key=lambda r: (r["threshold"], -r["patience"]))]
    def val_recall(r: dict) -> float:
        return (r["val_recall_car_10db"] + r["val_recall_clean"]) / 2

    top = max(val_recall(r) for r in eligible)
    near_top = [r for r in eligible if val_recall(r) >= top - RECALL_TOLERANCE]
    best = max(near_top, key=lambda r: (r["threshold"], -r["patience"]))
    thr, pat = best["threshold"], best["patience"]
    log.info(
        "chosen on validation data: threshold=%.3f patience=%d (val recall car10 %.3f, clean %.3f, "
        "val-features FA/h %.3f)",
        thr,
        pat,
        best["val_recall_car_10db"],
        best["val_recall_clean"],
        best["val_features_fa_per_hour"],
    )

    # ---- test metrics at the chosen operating point --------------------------------------------
    recall = {name: summarize_recall(utt, groups[f"test_pos/{name}"], thr, pat) for name in conditions}
    recall["command_clean"] = summarize_recall(utt, groups["test_cmd/clean"], thr, pat)
    recall["command_car_10db"] = summarize_recall(utt, groups["test_cmd/car_10db"], thr, pat)

    # Supplemental: recall on test positives that Whisper transcribes as "... drive ... pilot"
    # (check_tts.py --set test/positive --n 0), i.e. excluding TTS renditions that are mispronounced.
    asr_verified = None
    tts_check = MODELS_DIR / "tts_check.json"
    if tts_check.exists():
        verdicts = json.loads(tts_check.read_text(encoding="utf-8")).get("test/positive", {}).get("clips", {})
        test_paths = [r["path"] for r in test_pos]
        if verdicts and sum(p in verdicts for p in test_paths) >= 0.95 * len(test_paths):
            ok_paths = {p for p in test_paths if verdicts.get(p)}
            asr_verified = {"fraction_of_test_positives_verified": round(len(ok_paths) / len(test_paths), 4)}
            for name in conditions:
                keys = [k for k in groups[f"test_pos/{name}"] if k.split("|", 1)[1] in ok_paths]
                asr_verified[name] = summarize_recall(utt, keys, thr, pat)

    def adversarial(group: str) -> dict:
        keys = groups[group]
        fa = summarize_fa(utt, keys, thr, pat)
        trig = [k for k in keys if len(fired_frames(utt[k]["scores"], thr, pat)) > 0]
        fa.update(
            {
                "clips": len(keys),
                "clips_triggered": len(trig),
                "triggered_frac": round(len(trig) / max(1, len(keys)), 4),
            }
        )
        return fa

    neg_rows = {r["path"]: r for r in test_neg}
    adv_clean = adversarial("test_adv/clean")
    adv_car = adversarial("test_adv/car_10db")
    triggered_texts: dict[str, int] = {}
    for k in groups["test_adv/clean"] + groups["test_adv/car_10db"]:
        if len(fired_frames(utt[k]["scores"], thr, pat)) > 0:
            t = neg_rows[k.split("|", 1)[1]]["text"]
            triggered_texts[t] = triggered_texts.get(t, 0) + 1
    false_accepts = {
        "validation_set_features": {
            **summarize_fa(val_feat, ["val_features"], thr, pat),
            "note": "used for checkpoint selection and threshold choice",
        },
        "librispeech_test_clean_other": summarize_fa(longs, libri_keys, thr, pat),
        "librispeech_test_clean_car_10db": summarize_fa(longs, libri_car_keys, thr, pat),
        "car_noise_only_demand_tcar": summarize_fa(longs, car_keys, thr, pat),
        "adversarial_test_clean": adv_clean,
        "adversarial_test_car_10db": adv_car,
    }
    false_accepts["adversarial_triggered_texts"] = dict(sorted(triggered_texts.items(), key=lambda kv: -kv[1])[:30])

    # per-voice recall in car noise 10 dB (test speakers)
    pos_rows = {r["path"]: r for r in test_pos}
    per_voice: dict[str, list[int]] = {}
    for k in groups["test_pos/car_10db"]:
        v = pos_rows[k.split("|", 1)[1]]["voice"]
        per_voice.setdefault(v, [0, 0])
        per_voice[v][0] += utterance_detected(utt[k], thr, pat)[0]
        per_voice[v][1] += 1
    per_voice_out = {v: {"n": n, "recall": round(h / n, 4)} for v, (h, n) in sorted(per_voice.items())}
    per_text: dict[str, list[int]] = {}
    for k in groups["test_pos/clean"]:
        t = pos_rows[k.split("|", 1)[1]]["text"]
        per_text.setdefault(t, [0, 0])
        per_text[t][0] += utterance_detected(utt[k], thr, pat)[0]
        per_text[t][1] += 1

    sec_per_chunk = np.median([r["sec_per_chunk"] for r in list(utt.values()) + list(longs.values())])
    librispeech_fa = false_accepts["librispeech_test_clean_other"]["per_hour"]
    metrics = {
        "model": (str(Path(model_path).relative_to(REPO_ROOT)) if Path(model_path).is_relative_to(REPO_ROOT)
                  else model_path).replace("\\", "/"),
        "runtime_api": (
            "openwakeword.Model(wakeword_models=[...], inference_framework='onnx').predict(1280-sample int16 chunk)"
        ),
        "quick_mode": bool(args.quick),
        "recommended": {
            "threshold": thr,
            "patience_frames": pat,
            "debounce_time_s": DEBOUNCE_S,
            "how": f"model.predict(chunk, threshold={{'{MODEL_NAME}': {thr}}}, debounce_time={DEBOUNCE_S})"
            if pat == 1
            else f"fire when the last {pat} raw scores are all >= {thr} (app-side), then ignore {DEBOUNCE_S} s",
            "selection": (
                f"highest threshold whose validation-speaker recall (mean of clean and car 10 dB) is within "
                f"{RECALL_TOLERANCE:.0%} of the best threshold with <= {FA_SELECTION_LIMIT} false accepts/h on "
                "validation_set_features.npy; test data not used"
            ),
        },
        "recall": recall,
        "false_accepts": false_accepts,
        "targets": TARGETS,
        "targets_met": {
            "recall_clean": recall["clean"]["recall"] >= TARGETS["recall_clean"],
            "recall_car_10db": recall["car_10db"]["recall"] >= TARGETS["recall_car_10db"],
            "false_accepts_per_hour_librispeech": (
                librispeech_fa is not None and librispeech_fa < TARGETS["false_accepts_per_hour"]
            ),
            "false_accepts_per_hour_validation_set": (
                false_accepts["validation_set_features"]["per_hour"] < TARGETS["false_accepts_per_hour"]
            ),
        },
        "recall_asr_verified_positives": asr_verified,
        "per_voice_recall_car_10db": per_voice_out,
        "per_text_recall_clean": {t: round(h / n, 4) for t, (h, n) in sorted(per_text.items())},
        "threshold_sweep": sweep,
        "eval_setup": {
            "lead_in_s": LEAD_S,
            "tail_s": TAIL_S,
            "detection_window": "clip start .. clip end + 1 s",
            "speech_rms_dbfs_range": SPEECH_RMS_DBFS,
            "snr_definition": "RMS of the utterance vs RMS of the noise",
            "car_noise": "DEMAND TCAR (16 ch, 16 kHz) - held out from training",
            "rirs": "MIT environmental IRs, test split (10%), held out from training",
            "debounce_s": DEBOUNCE_S,
            "librispeech_chapters": len(libri_keys),
        },
        "timing": {"median_predict_ms_per_80ms_chunk": round(1000 * float(sec_per_chunk), 3)},
        "runtime_env": runtime_env(),
    }
    write_json(Path(args.output), metrics)
    log.info("recall: %s", {k: v["recall"] for k, v in recall.items()})
    log.info(
        "false accepts/h: %s",
        {k: v.get("per_hour") for k, v in false_accepts.items() if isinstance(v, dict) and "per_hour" in v},
    )
    log.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
