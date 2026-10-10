# "Hey DrivePilot" wake word (openWakeWord)

Custom [openWakeWord](https://github.com/dscripka/openWakeWord) model for the phrase **"hey drive pilot"**,
trained entirely on synthetic speech plus real noise, with openWakeWord's own feature models and training code.

* Model: `models/wakeword/hey_drivepilot.onnx` (tracked in git, 205 KB; rebuild with the commands below)
* Metrics: `models/wakeword/metrics.json` (written by `evaluate.py`)
* Training log/history: `models/wakeword/training_history.json`, `data/wakeword/logs/*.log`

## Using it in the app

```python
import openwakeword

model = openwakeword.Model(wakeword_models=["models/wakeword/hey_drivepilot.onnx"], inference_framework="onnx")
# feed 16 kHz mono int16 audio in 1280-sample (80 ms) chunks
scores = model.predict(chunk, threshold={"hey_drivepilot": 0.25}, debounce_time=2.0)
if scores["hey_drivepilot"] >= 0.25:
    ...  # wake word detected
```

0.25 is `recommended.threshold` in `metrics.json` (see Results). The feature models
(`melspectrogram.onnx`, `embedding_model.onnx`) must be present in the openwakeword package; run
`openwakeword.utils.download_models(["__feature_models_only__"])` once if they are missing.

Do **not** use `Model.predict(..., patience=...)` in openwakeword 0.6.0: it checks the previous frames in the
prediction buffer *after* they have been zeroed by the same patience rule, so it never fires once a frame
is suppressed. `debounce_time` works as documented.

## Results

All numbers below come from `evaluate.py` (`models/wakeword/metrics.json`): audio from **held-out** voices and
speakers is streamed through `openwakeword.Model(...).predict()` in 80 ms int16 chunks. The operating point is
chosen on validation data only (validation speakers + `validation_set_features.npy`): the highest threshold whose
validation recall is within 1 point of the best threshold with <= 0.25 FA/h.

**Recommended setting: `threshold = 0.25`, `debounce_time = 2.0` s, no patience.**

| Recall on held-out test voices (threshold 0.25) | n | recall | 95% CI | target |
| --- | --- | --- | --- | --- |
| clean (zero padding, like `predict_clip`) | 3000 | **97.1%** | 96.5-97.7% | >= 95%: met |
| real car-interior noise, 10 dB SNR (DEMAND TCAR, a car never used in training) | 3000 | **97.5%** | 96.8-98.0% | >= 90%: met |
| real car-interior noise, 5 dB SNR | 3000 | **97.4%** | 96.8-97.9% | - |
| car noise 10 dB + held-out room impulse responses | 3000 | 92.7% | 91.7-93.6% | - |
| wake phrase directly followed by a command ("hey drive pilot, can I go?"), clean | 600 | 97.3% | 95.7-98.4% | - |
| same, car noise 10 dB | 600 | 96.3% | 94.5-97.6% | - |

| False accepts (threshold 0.25, 2 s debounce) | hours | false accepts | per hour |
| --- | --- | --- | --- |
| `validation_set_features.npy` (DiPCo, Santa Barbara corpus, MUSDB; also used for selection) | 10.7 | 0 | **0.0** |
| LibriSpeech test-clean + test-other, streamed (independent) | 10.7 | 0 | **0.0** |
| LibriSpeech test-clean + DEMAND car noise at 10 dB | 5.4 | 0 | **0.0** |
| DEMAND car-interior noise alone | 1.3 | 0 | **0.0** |
| held-out adversarial near-miss phrases (3000 clips, unseen speakers and texts), clean | 3.9 | 9 | 2.3 (0.30% of clips) |
| same, car noise 10 dB | 3.9 | 15 | 3.8 (0.50% of clips) |

Zero events in 28 h of general negative audio bounds the true rate below ~0.11/h (95% upper bound over all four
sets), or ~0.18/h using only the 17.4 h of independent streamed audio. The adversarial set is dense near-miss
speech, so its per-hour number is not comparable with the others; the phrases that trigger are mostly
"hey private pilot", "... drive pilot" inside longer phrases, "hey my pilot" and "hey drive by".

Other operating points (same test data; the validation side is in `metrics.json["threshold_sweep"]`):

| threshold | clean | car 10 dB | car 5 dB | car 10 dB + reverb | adversarial clips triggered | general FA/h |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 98.1% | 98.2% | 98.4% | 94.5% | 0.43% | 0 |
| **0.25** | **97.1%** | **97.5%** | **97.4%** | **92.7%** | **0.30%** | **0** |
| 0.50 | 96.2% | 96.2% | 96.2% | 90.3% | 0.20% | 0 |
| 0.70 | 94.9% | 94.9% | 95.2% | 88.2% | 0.17% | 0 |
| 0.90 | 92.3% | 92.1% | 92.2% | 84.0% | 0.13% | 0 |

Patience (k consecutive frames over the threshold) never helped: it lowers recall, and the false-accept sets are
already at zero.

More in `metrics.json`: per-voice recall (lowest: `en_US-reza_ibrahim` 86.9%, `en_US-joe` 93.7%,
`en_US-libritts-high` 94.0%; the others 97-100%), per-text-variant recall, and recall on the 85.9% of test
positives that Whisper transcribes as "... drive ... pilot" (`check_tts.py`): 98.2% clean, 98.3% at 10 dB, 98.2%
at 5 dB, so a good part of the misses are TTS renditions that do not clearly say the phrase. Streaming cost: median
11 ms of CPU per 80 ms chunk, measured with 12 evaluation processes in parallel on the i7-13700H.

## Reproduce

Two virtual environments:

* `.venv` (the app's, never modified): runs `evaluate.py` and `check_tts.py` with openwakeword 0.6.0 +
  onnxruntime 1.31.0 (CPU), i.e. exactly the runtime the app uses.
* `.venv-wakeword` (training only, gitignored): openWakeWord's 2022-era training stack does not install on
  current packages as-is (e.g. `acoustics` needs `scipy<1.15`; onnxruntime-gpu 1.31 needs CUDA 13), so:

```bash
uv venv --python 3.11 .venv-wakeword
uv pip install --python .venv-wakeword/Scripts/python.exe "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-wakeword/Scripts/python.exe "numpy<2.3" "scipy==1.14.1" soundfile soxr tqdm pyyaml requests scikit-learn "onnxruntime-gpu==1.22.0" onnx audiomentations torch-audiomentations speechbrain mutagen acoustics pronouncing torchinfo torchmetrics pyarrow psutil pathvalidate
uv pip install --python .venv-wakeword/Scripts/python.exe --no-deps "openwakeword==0.6.0" "piper-tts==1.8.0"
uv pip install --python .venv-wakeword/Scripts/python.exe ruff pytest   # optional: lint + unit tests
.venv-wakeword/Scripts/python -c "import openwakeword.utils as u; u.download_models(['__feature_models_only__'])"
```

(Tested versions: torch 2.6.0+cu126, torchaudio 2.6.0, onnxruntime-gpu 1.22.0, numpy 2.2.6, scipy 1.14.1,
audiomentations 0.43.1, torch-audiomentations 0.12.0, speechbrain 1.1.1, piper-tts 1.8.0, openwakeword 0.6.0.)

Then, from the repo root (`PY=.venv-wakeword/Scripts/python`):

```bash
$PY scripts/wakeword/download_data.py --all          # ~25 GB download, see "Data" below
$PY scripts/wakeword/prepare_noise.py                # noise pools -> data/wakeword/noise/
$PY scripts/wakeword/generate_clips.py --all         # Piper TTS clips -> data/wakeword/clips/ (~1.5 h on 12 CPU workers)
$PY scripts/wakeword/augment_features.py --all       # augmentation + openWakeWord features (GPU)
$PY scripts/wakeword/train_model.py                  # training (GPU) + ONNX export -> models/wakeword/
.venv/Scripts/python scripts/wakeword/evaluate.py    # streaming evaluation -> models/wakeword/metrics.json
.venv/Scripts/python scripts/wakeword/check_tts.py --set test/positive --n 0   # optional, before evaluate.py
.venv-wakeword/Scripts/python -m pytest scripts/wakeword -q   # unit tests (splits, mixing, detection rule)
```

`check_tts.py` adds the Whisper check behind the ASR-verified recall in `metrics.json`. `evaluate.py` takes about
1 h on 12 CPU processes; it caches raw scores in `data/wakeword/eval_scores.pkl`, and `--from-scores`
recomputes the metrics (e.g. after changing the threshold rule) in a minute.
`train_model.py --override layer_size=64 --output <dir>/hey_drivepilot.onnx` trains a variant without editing
the config. Measured on this laptop while it was shared with other work: synthesis ~1.5 h, augmentation ~45 min,
training 52 min (data-loading bound, ~20 steps/s), evaluation ~1 h.

Every step is resumable/idempotent (existing outputs are skipped; `--overwrite` where relevant).
Hyperparameters are in `training_config.yaml`; data splits, voices and paths in `common.py`.

## Scripts

| File | Purpose |
| --- | --- |
| `common.py` | Paths, audio constants, Piper voice list and the train/val/test voice + speaker split |
| `download_data.py` | Downloads and verifies (sha256/md5) all external data |
| `prepare_noise.py` | Builds 16 kHz int16 noise pools: `train_car`, `train_general`, `test_car` |
| `phrases.py` | Positive text variants, near-miss / in-car / adversarial negative texts, held-out text split |
| `generate_clips.py` | Piper synthesis of positives/negatives per split (random speaker, length/noise scales) |
| `augment_features.py` | Reverb, noise (car + general), EQ, distortion, pitch, gain; openWakeWord features |
| `train_model.py` | `openwakeword.train.Model.auto_train` on the GPU, ONNX export + export check |
| `training_config.yaml` | Training hyperparameters |
| `evaluate.py` | Streams audio through `openwakeword.Model` in 80 ms chunks; recall / false accepts / threshold |
| `check_tts.py` | Whisper check that the synthetic positives really say "drive pilot" |
| `test_wakeword_tools.py` | Unit tests: split disjointness, window placement, SNR mixing, detection rule |

## Method

1. **Positives.** Text variants `hey drive pilot`, `hey, drive pilot`, `hey drive-pilot`, `Hey DrivePilot`,
   `Hey Drive Pilot!`, `Hey, Drive Pilot.`, `hey drive pilot?`, `Hey, DrivePilot!` (espeak-ng reads the single
   token "drivepilot" as "drive-pih-LOT", so the joined spelling is written in camel case or with a hyphen).
   Each clip: random voice and speaker of its split, `length_scale` = voice default x U(0.75, 1.30),
   `noise_scale`, `noise_w` ~ U(0.333, 0.98); silence trimmed, resampled to 16 kHz mono 16-bit.
2. **Voice / speaker split** (`common.py`, written to `data/wakeword/clips/speaker_splits.json`):
   multi-speaker voices are split by speaker with a fixed seed (12% test, 6% validation, rest train;
   `en_US-libritts-high` uses the same reader split as `en_US-libritts_r-medium`), and whole single-speaker
   voices are held out: test = `en_GB-alan`, `en_US-joe`, `en_US-kristin`, `en_GB-northern_english_male`,
   `en_US-hfc_female`, `en_US-reza_ibrahim`; validation = `en_US-john`, `en_GB-jenny_dioco`.
   `en_US-l2arctic-medium` is excluded from all positive sets (see Known issues) but used for negatives.
3. **Adversarial negatives** (same synthesis): hand-written near misses (`hey driver`, `drive pilot`,
   `hey pilot`, `hey drive`, `drive by`, `hey private`, `hey dry pile`, `hey co-pilot`, other assistants' wake
   words, ...), ordinary in-car speech (`turn left here`, `what's the speed limit`, `hey can you drive`, ...)
   and phonetically similar phrases from `openwakeword.data.generate_adversarial_texts` (homophones of the
   target removed — the library's own homophone filter never matches). 15% of every text list is held out
   for the test set only.
4. **Augmentation** (`augment_features.py`): 2 rounds per training clip; MIT impulse responses (p=0.5,
   train split of the IRs), right-aligned placement in a 2 s window with 0-200 ms end jitter, 7-band EQ and
   tanh distortion (p=0.25), background noise per example — real car interior (QUT-NOISE CAR) at 0-20 dB SNR
   (40%), general background (DEMAND environments + AudioSet) at -3-15 dB (30%), both at 5-20 dB (10%),
   none (20%) — then pitch shift +-3 st, band-stop, coloured noise 10-30 dB (p=0.25 each) and gain -24..0 dB on
   the GPU. Plus 20k noise-only windows as extra negatives.
5. **Features**: openWakeWord's frozen `melspectrogram.onnx` + `embedding_model.onnx` (Google
   speech_embedding), computed on the GPU with TF32 off (matches CPU inference to ~1e-4).
6. **Training**: unchanged `openwakeword.train.Model.auto_train` (DNN, 3 sequences, hard-example mining,
   negative weight ramp to `max_negative_weight`, checkpoint averaging), batches from ACAV100M features
   + positives + adversarial negatives + noise-only negatives via `mmap_batch_generator`; checkpoint
   selection on validation speakers and `validation_set_features.npy`.
7. **Evaluation** (`evaluate.py`, main venv): see its docstring. Everything is streamed through
   `openwakeword.Model.predict` in 80 ms chunks with a 2 s lead-in and 1.5 s tail per utterance; a detection
   counts from the start of the utterance to 1 s after its end; SNR = utterance RMS vs noise RMS; speech level
   random in -32..-20 dBFS RMS.

Data volumes: 30,000 positive and 24,000 adversarial training clips synthesized (27,125 positives used after the
exclusions), each augmented twice: 54,250 positive and 48,000 adversarial feature windows, plus 20,000 noise-only
windows and the 5.6 M ACAV100M windows. Validation: 1,945 positive + 2,000 negative clips. Test: 3,000 positive,
600 phrase+command and 3,000 adversarial clips. Noise: 3.0 h training car noise (QUT-NOISE CAR), 5.5 h general
background (DEMAND + AudioSet), 1.3 h held-out car noise (DEMAND TCAR, 16 channels).

## Data and licenses

| Data | Source | License | Use |
| --- | --- | --- | --- |
| openWakeWord feature models | github.com/dscripka/openWakeWord releases v0.5.1 | Apache-2.0 | frozen features |
| ACAV100M 2000 h features, `validation_set_features.npy` | HF `davidscripka/openwakeword_features` | **CC BY-NC-SA 4.0** | general negatives; FA validation |
| Piper voices (30) | HF `rhasspy/piper-voices` | per voice (below) | positive/negative synthesis |
| MIT environmental impulse responses | HF `davidscripka/MIT_environmental_impulse_responses` (MIT IR Survey, Traer & McDermott 2016) | unknown per the HF card; cite the source | reverb (train/test split by file) |
| QUT-NOISE CAR | QUT Research Data Finder | CC BY-SA 3.0 | training car noise |
| DEMAND (TCAR + 8 other environments) | Zenodo record 1227121 | CC BY 4.0 | TCAR = held-out test car noise; others = training background |
| AudioSet balanced train, 3 parquet shards (1500 clips) | HF `agkphysics/AudioSet` | CC BY 4.0 (labels; audio from YouTube) | training background |
| LibriSpeech test-clean + test-other | OpenSLR 12 | CC BY 4.0 | false-accept test |

Piper voice licenses (from each voice's MODEL_CARD): libritts_r, libritts-high, vctk, aru, alba: CC BY 4.0;
ljspeech, bryce, norman, cori, john, kristin: public domain; kathleen, mike, joe, reza_ibrahim: CC0;
sam: Apache-2.0; southern/northern_english: CC BY-SA 4.0; l2arctic: CC BY-NC 4.0; semaine, ryan, hfc_male,
hfc_female: CC BY-NC-SA 4.0; lessac: Blizzard 2013 license (non-commercial); arctic: CMU ARCTIC license;
amy, danny, kusal, alan, jenny_dioco: see the linked dataset pages.

**Because the ACAV100M features and several voices are non-commercial (NC) licensed, this model is
suitable for the hackathon / non-commercial use only.** A commercial build would need to retrain without
the NC data.

## Known issues / compromises

* **Synthetic speech only.** Every positive (train and test) is Piper TTS; no real recordings of "hey drive pilot"
  were available. Recall on real voices through the laptop's far-field mics may be lower than the synthetic
  test numbers. Before the demo, record 20-50 real utterances (several people, in the car) and check that the
  scores clear 0.25; if real recall is poor, lower the threshold (0.10 still had zero general false accepts).
* **TTS quality.** Whisper small.en recognises "drive ... pilot" in ~85% of the positives (train sample 84.7%,
  test 85.9%); the rest are slurred or mispronounced renditions. They were kept, as in openWakeWord's own
  pipeline. `en_US-l2arctic-medium` was dropped from all positive sets because only 7.5% of its renditions were
  recognised (it is still used for negatives); this removes the L2-accented voices from the positives.
* **Held-out car noise is one car.** The car conditions use DEMAND TCAR (16 mics, 5 min, one car); training car
  noise is QUT-NOISE CAR (a different car and recordings). Neither is the team's car or laptop microphone.
* **False-accept sets.** `validation_set_features.npy` was used for checkpoint selection and threshold choice, so
  its 0/h is not independent; LibriSpeech (read speech) is the independent streamed test. There is no real
  in-car conversation or radio recording among the false-accept tests.
* **Threshold rule.** A first evaluation pass used the rule "maximum validation recall", which picked 0.10. Before
  finalising, the rule was changed to "highest threshold within 1 point of the best validation recall", to keep a
  margin above 0.05 where validation false accepts start (0.28/h); that gives 0.25. Both rules use validation data
  only and both meet every target on test (see the sweep table).
* **Data deviations.** `train/negative` was reduced from 32k to 24k clips for time. The l2arctic exclusion was
  added while `train/positive` was being generated, so the voice mix of the training positives differs from the
  configured weights (more vctk/arctic/aru, fewer libritts_r); a fresh run gives a statistically equivalent, not
  bit-identical, set. 538 positives longer than 2.2 s (~2%) were not used for training (their "hey" would fall
  outside the 2 s window and look like the negative "drive pilot"). 238 of the 270 MIT impulse responses are used
  for training and 32 are held out for the reverb test condition.
* **Pipeline adaptations** (method otherwise unchanged): openWakeWord's training stack needs an old pinned
  environment (`.venv-wakeword`, see Reproduce). Augmentation mixes noise per example with the SNR measured on the
  speech, adds the car-noise stage and noise-only negatives, applies reverb before noise, widens gain to -24 dB and
  runs torch-audiomentations in sub-batches of 16 (PitchShift on 128-clip batches exhausted GPU and host memory).
  `train.py`'s 3 GB false-positive window tensor is built lazily on the GPU, and batches come from an in-process
  loader. `auto_train` hard-codes 11.3 h for the validation set (actual: 10.7 h); that only affects its internal
  checkpoint selection, and every reported FA/h uses the real durations.
* **openwakeword 0.6.0 `patience` bug** (see "Using it in the app"); use `debounce_time` instead.
* **No TFLite export** (only ONNX was needed; openWakeWord's converter needs TensorFlow + onnx-tf).
* **Evaluation environment.** The final `evaluate.py` run used `.venv-wakeword` with the app's versions of
  openwakeword (0.6.0) and onnxruntime (1.31.0, CPU), because the app venv was being repaired at the time. The
  model was then checked in `.venv` and gave bit-identical scores on the same clips.
* **Licenses**: see above; the model inherits non-commercial terms from the ACAV100M features and several voices.

## Disk usage and cleanup

`data/wakeword/` holds about 34 GB (features 17.5, downloads incl. extracted noise 9.0, clips 2.7, voices 2.1, noise pools 1.1, oww_features 0.8, LibriSpeech 0.7), of which 17.3 GB is the ACAV100M feature file (needed only for
training). All of it can be deleted once the model is final; `models/wakeword/` (~2 MB) is all the app needs.
`.venv-wakeword/` (about 5.7 GB) is only needed to retrain.
