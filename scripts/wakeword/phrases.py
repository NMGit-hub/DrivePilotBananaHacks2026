"""Text for positive and negative clips of the "Hey DrivePilot" wake word.

Positive texts are spelling/punctuation variants that make Piper (espeak-ng) produce different
prosody for the same phrase. Note: espeak-ng mispronounces the single token "drivepilot" as
"drive-pih-LOT", so the joined form is written "Hey DrivePilot" (espeak splits the camel case) and
"hey drive-pilot" (no pause between the words) instead.

Negative texts:
  * hand-written near misses (partial phrases, rhymes, other assistants' wake words),
  * ordinary in-car speech (navigation, small talk, questions to the co-pilot without the wake word),
  * phonetically similar phrases from ``openwakeword.data.generate_adversarial_texts`` (the method of
    the official openWakeWord training pipeline), post-filtered to drop homophones of the target.

A fixed subset of every negative list is reserved for the held-out adversarial test set.
"""

from __future__ import annotations

import re

import numpy as np
from common import stable_seed

TARGET_PHRASE = "hey drive pilot"

# (text, weight) — weights are relative sampling frequencies.
POSITIVE_TEXTS: list[tuple[str, float]] = [
    ("hey drive pilot", 0.30),
    ("Hey DrivePilot", 0.10),
    ("hey, drive pilot", 0.18),
    ("hey drive-pilot", 0.14),
    ("Hey Drive Pilot!", 0.12),
    ("Hey, Drive Pilot.", 0.08),
    ("hey drive pilot?", 0.04),
    ("Hey, DrivePilot!", 0.04),
]

# Test-only: the wake phrase followed directly by a command, as it will be used in the car.
POSITIVE_WITH_COMMAND_TEXTS: list[str] = [
    "hey drive pilot, what's the speed limit?",
    "hey drive pilot, can I go?",
    "hey drive pilot, who got here first?",
    "hey drive pilot, which exit do I take?",
    "hey drive pilot, repeat that.",
    "hey drive pilot, quiet mode.",
    "Hey DrivePilot, I'm going to the airport.",
    "hey drive pilot, which lane for five twenty east?",
    "Hey Drive Pilot, can I turn right on red here?",
    "hey, drive pilot, stop.",
]

NEAR_MISS_TEXTS: list[str] = [
    "hey driver",
    "hey drivers",
    "drive pilot",
    "drive pilot mode",
    "hey pilot",
    "hey drive",
    "drive by",
    "hey private",
    "hey dry pile",
    "hey drive by",
    "hey diver",
    "hey driving",
    "hey dried pie lot",
    "hey co-pilot",
    "hey copilot",
    "hey autopilot",
    "autopilot",
    "co-pilot",
    "hi pilot",
    "hey Pilate",
    "hey the pilot",
    "hey my pilot",
    "they drive",
    "they drive pilots",
    "pay the pilot",
    "hey drive slow",
    "hey drive safe",
    "hey drive time",
    "a drive-by",
    "hey thrive",
    "hey five pilots",
    "hey drive in",
    "hey private pilot",
    "private pilot",
    "hey dry eye",
    "hey drywall",
    "hey Dr. Pilot",
    "hey drive light",
    "hey Siri",
    "hey Google",
    "okay Google",
    "Alexa",
    "hey Jarvis",
    "hey Cortana",
    "hey Mycroft",
    "hey Rhasspy",
    "hey there",
    "hey you",
    "hey dad",
    "hey mom",
    "hey look",
    "hey what's up",
    "hey buddy",
    "hey guys",
    "hi there",
    "hello",
    "drive",
    "pilot",
    "hey",
    "the driver's pilot",
    "high pilot",
    "hey Dave",
    "hey drive pile",
    "hey dry Pyle",
    "hay ride",
    "hey ride along",
    "hey ride pilot",
    "hey drive pilaf",
    "hey drive island",
    "hey drive Pilates",
    "hey pilots",
    "drive pylon",
    "Hey Tribe",
    "hey rifle",
    "hey Irvine",
    "hey drive a pickup",
    "hey drive past",
    "hey derive",
    "hey arrive",
    "hey alive",
]

IN_CAR_TEXTS: list[str] = [
    "turn left here",
    "turn right at the light",
    "what's the speed limit",
    "hey can you drive",
    "can I go now",
    "who got here first",
    "which exit do I take",
    "how far to the next gas station",
    "is it my turn",
    "I'll drive today",
    "let's go for a drive",
    "the pilot said we're landing soon",
    "slow down please",
    "watch out for that cyclist",
    "pull over here",
    "take the next exit",
    "merge left",
    "we're almost there",
    "what lane should I be in",
    "play some music",
    "call mom",
    "navigate home",
    "stop at the light",
    "hey watch out",
    "hey look at that",
    "hey slow down",
    "hey turn left",
    "drive safe",
    "drive slowly",
    "the driver in front of us is texting",
    "my private pilot license expired",
    "the pilot light is out",
    "let's get a drive-through coffee",
    "park in the driveway",
    "my hard drive is full",
    "I took it for a test drive",
    "it's a pilot program",
    "can you be my co-pilot",
    "the car on the right goes first",
    "the white SUV was first",
    "you're up, check and go",
    "in half a mile move one lane left",
    "is the express lane open",
    "how long until we get there",
    "I'm hungry, let's stop for food",
    "did you see that sign",
    "no turn on red until seven",
    "keep going straight",
    "make a U-turn when possible",
    "there's traffic ahead",
    "roll down the window",
    "turn up the air conditioning",
    "what time is it",
    "text Sarah that I'm running late",
    "who's driving tomorrow",
    "the drive was beautiful",
    "pilots and drivers both need rest",
    "I want to drive to the coast",
    "hey, can you hand me that",
    "hey, did you lock the door",
    "hey, what was that noise",
    "a pilot and a driver walked into a bar",
    "she drives a pickup",
    "drive past the school",
    "the private road is closed",
    "hey, it's raining again",
    "high five",
    "hey, take the highway",
    "I think we missed the exit",
    "drive five more miles",
    "they fly private",
    "the river is high today",
    "we arrived at five",
    "hey, pile it in the back",
    "hey, try the pilot app",
    "turn the radio down",
    "honk if you see them",
    "the light is green",
    "that driver cut me off",
    "hey, find a parking spot",
    "drive carefully in the rain",
]

# Fraction of each negative list reserved for the held-out test set (by stable hash of the text).
TEST_TEXT_FRACTION = 0.15

_TARGET_PHONES = None


def _strip_stress(phones: str) -> str:
    return re.sub(r"\d", "", phones)


def _target_phone_string() -> str:
    import pronouncing

    global _TARGET_PHONES
    if _TARGET_PHONES is None:
        _TARGET_PHONES = " ".join(_strip_stress(pronouncing.phones_for_word(w)[0]) for w in TARGET_PHRASE.split())
    return _TARGET_PHONES


def sounds_like_target(text: str) -> bool:
    """True if any CMUdict pronunciation of ``text`` contains the target phoneme sequence.

    openwakeword's ``generate_adversarial_texts`` intends to exclude homophones, but its check compares
    a string with a list and never matches, so e.g. "hay drive pilot" can be produced. Such texts
    would be mislabeled negatives, so they are dropped here.
    """
    import itertools

    import pronouncing

    words = re.findall(r"[a-zA-Z']+", text.lower())
    options = []
    for w in words:
        prons = pronouncing.phones_for_word(w)
        if not prons:
            return False
        options.append([_strip_stress(p) for p in prons[:3]])
    target = _target_phone_string()
    return any(target in " ".join(combo) for combo in itertools.product(*options))


def is_test_text(text: str) -> bool:
    """Deterministically assign a negative text to the held-out test set."""
    return (stable_seed("negtext", text.lower()) % 1000) < TEST_TEXT_FRACTION * 1000


def adversarial_texts(n: int, seed: int = 0) -> list[str]:
    """Phonetically similar phrases via openWakeWord's generator, homophones removed, deduplicated."""
    from openwakeword.data import generate_adversarial_texts

    np.random.seed(seed)
    texts = generate_adversarial_texts(
        input_text=TARGET_PHRASE, N=n, include_partial_phrase=1.0, include_input_words=0.2
    )
    out = sorted({t for t in texts if t.strip() and not sounds_like_target(t)})
    return out


def negative_text_pools(n_adversarial: int = 20000) -> dict[str, dict[str, list[str]]]:
    """Return ``{"train"|"test": {"near_miss": [...], "in_car": [...], "adversarial": [...]}}``."""
    pools: dict[str, dict[str, list[str]]] = {"train": {}, "test": {}}
    sources = {
        "near_miss": NEAR_MISS_TEXTS,
        "in_car": IN_CAR_TEXTS,
        "adversarial": adversarial_texts(n_adversarial),
    }
    for name, texts in sources.items():
        pools["train"][name] = [t for t in texts if not is_test_text(t)]
        pools["test"][name] = [t for t in texts if is_test_text(t)]
    return pools
