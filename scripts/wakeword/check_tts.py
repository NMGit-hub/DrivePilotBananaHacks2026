"""Quality check: are the synthesized positives actually intelligible as "hey drive pilot"?

Transcribes a random sample of positive clips with faster-whisper (main venv) and reports, per Piper
voice, the fraction whose transcript contains both "drive" and "pilot" (also accepting the spellings
"drivepilot"/"drive-pilot"). This catches voices/parameter ranges that produce mumbled or wrong
audio. It is a sanity check on the training data, not a model metric.

    .venv/Scripts/python scripts/wakeword/check_tts.py --set train/positive --n 400
    .venv/Scripts/python scripts/wakeword/check_tts.py --set test/positive --n 0   # all test clips

With ``--n 0`` on test/positive, evaluate.py additionally reports recall on the ASR-verified subset.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re

from common import CLIPS_DIR, MODELS_DIR, POSITIVE_EXCLUDED_VOICES, setup_logging, write_json

log = logging.getLogger("check_tts")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="sets", nargs="+", default=["train/positive", "test/positive"])
    parser.add_argument("--n", type=int, default=400, help="clips sampled per set (0 = all)")
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--include-excluded", action="store_true",
                        help="also check voices in POSITIVE_EXCLUDED_VOICES (not used as positives)")
    args = parser.parse_args()
    setup_logging("check_tts")

    # Import torch first: it puts its bundled cuBLAS/cuDNN DLLs on the search path CTranslate2 needs.
    import torch  # noqa: F401
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    report = {}
    for set_name in args.sets:
        lines = (CLIPS_DIR / set_name / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
        rows = [r for r in {r["path"]: r for r in rows}.values() if (CLIPS_DIR / r["path"]).exists()]
        if not args.include_excluded:
            rows = [r for r in rows if r["voice"] not in POSITIVE_EXCLUDED_VOICES]
        random.Random(0).shuffle(rows)
        per_voice: dict[str, list[int]] = {}
        bad_examples = []
        verdicts: dict[str, bool] = {}
        for r in rows[: args.n or None]:
            segments, _ = model.transcribe(
                str(CLIPS_DIR / r["path"]), language="en", beam_size=5, condition_on_previous_text=False
            )
            text = " ".join(s.text for s in segments).lower()
            norm = re.sub(r"[^a-z ]", " ", text.replace("-", " "))
            ok = ("drive" in norm and "pilot" in norm) or "drivepilot" in norm.replace(" ", "")
            verdicts[r["path"]] = bool(ok)
            v = per_voice.setdefault(r["voice"], [0, 0])
            v[0] += ok
            v[1] += 1
            if not ok and len(bad_examples) < 25:
                bad_examples.append(
                    {
                        "path": r["path"],
                        "text": r["text"],
                        "heard": text.strip(),
                        "speaker": r["speaker_id"],
                        "length_scale": r["length_scale"],
                    }
                )
        total_ok = sum(v[0] for v in per_voice.values())
        total = sum(v[1] for v in per_voice.values())
        report[set_name] = {
            "n": total,
            "intelligible_frac": round(total_ok / max(1, total), 4),
            "per_voice": {k: {"n": n, "intelligible_frac": round(o / n, 3)} for k, (o, n) in sorted(per_voice.items())},
            "examples_not_recognized": bad_examples,
            "clips": verdicts,
        }
        log.info("%s: %d/%d transcripts contain 'drive' and 'pilot'", set_name, total_ok, total)
    out = MODELS_DIR / "tts_check.json"
    if out.exists():  # keep results for sets not checked in this run
        report = {**json.loads(out.read_text(encoding="utf-8")), **report}
    write_json(out, report)


if __name__ == "__main__":
    main()
