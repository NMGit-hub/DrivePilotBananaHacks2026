"""Question router: decides which handler answers a transcribed question (PLAN.md "How questions are routed").

A fast keyword/regex router, so routing costs microseconds and local answers stay instant:

- **control**: quiet mode, normal mode, repeat that, stop, never mind (voice controls);
- **destination**: "I'm going to <place>" (navigation by voice);
- **go watch / light watch**: "tell me when it's green", "let me know when I can go";
- **gap check**: split-second judgments ("can I merge now?", "is my blind spot clear?"): refused while moving;
- **right of way** (local, instant): "who got here first?", "can I go?", "how many cars are ahead of me?";
- **nav**: route questions ("did I miss my turn?", "which exit?", "how far to my turn?");
- **lane**: "which lane for 520 east?", "am I in the right lane?";
- **model**: everything else (signs, parking, lights, road awareness) goes to the vision model.

Rules are checked in a fixed order and the first match wins; the order encodes the tricky cases, e.g. "can I go
right on red?" is a sign question (model), not right of way, and "did I miss my turn?" is navigation while
"is it my turn?" is right of way. The router is pure (text in, `Route` out) and has a large table test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class RouteKind(StrEnum):
    """Which handler answers."""

    EMPTY = "empty"  # nothing but the wake word / filler
    CONTROL = "control"
    DESTINATION = "destination"
    WATCH = "watch"
    GAP_CHECK = "gap_check"
    RIGHT_OF_WAY = "right_of_way"
    NAV = "nav"
    LANE = "lane"
    MODEL = "model"


@dataclass(frozen=True, slots=True)
class Route:
    """The routing decision for one question."""

    kind: RouteKind
    intent: str  # handler-specific, e.g. "can_i_go", "quiet", "missed_turn"
    text: str  # the normalized question (for matching)
    question: str = ""  # the question as spoken, minus the wake word (what the model is asked)
    place: str | None = None  # destination for RouteKind.DESTINATION


_WAKE = re.compile(r"^\s*(?:(?:hey|hi|okay|ok|a)[\s,]+)?drive[\s-]*pilots?\b[\s,.!?:;-]*", re.IGNORECASE)
_CONTRACTIONS = {
    "what's": "what is",
    "who's": "who is",
    "whos": "who is",
    "it's": "it is",
    "where's": "where is",
    "how's": "how is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "i'm": "i am",
    "im": "i am",
    "let's": "let us",
    "lets": "let us",
    "can't": "can not",
    "cannot": "can not",
    "don't": "do not",
    "isn't": "is not",
    "aren't": "are not",
    "didn't": "did not",
    "i've": "i have",
    "i'll": "i will",
    "we're": "we are",
    "you're": "you are",
    "when's": "when is",
}
_FILLER = re.compile(r"\b(?:please|um+|uh+|erm|hmm+|just|actually|hey)\b|^(?:so|okay|ok|well|and|alright|right)\b")
_CONTRACTION = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b")


def strip_wake_word(text: str) -> str:
    """The question without a leading wake word ("Hey DrivePilot, ...")."""
    return _WAKE.sub("", text.replace("’", "'").replace("‘", "'").strip()).strip()


def normalize(text: str) -> str:
    """Lowercase, drop the wake word and filler words, expand common contractions, strip punctuation."""
    text = strip_wake_word(text).lower()
    text = _CONTRACTION.sub(lambda m: _CONTRACTIONS[m.group(1)], text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = _FILLER.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" '")


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# ---------------------------------------------------------------------------------------------- rules

_CONTROLS: list[tuple[str, re.Pattern[str]]] = [
    (
        "quiet",
        _rx(
            r"^(?:turn on |switch to |go to |go into |enter |use )?(?:quiet|silent) mode(?: on| now)?$",
            r"^(?:mute|silence) (?:the )?alerts$",
            r"^go quiet$",
        ),
    ),
    (
        "normal",
        _rx(
            r"^(?:turn on |switch to |go to |go back to |back to |use )?(?:normal|regular|loud) mode(?: on| now)?$",
            r"^(?:unmute|turn on) (?:the )?alerts$",
            r"^(?:quiet|silent) mode off$",
            r"^alerts on$",
        ),
    ),
    (
        "repeat",
        _rx(
            r"^(?:repeat|say) (?:that|it|again|that again|it again|the last thing)(?: again)?$",
            r"^repeat$",
            r"^what did you (?:just )?say$",
            r"^come again$",
            r"^pardon(?: me)?$",
            r"^sorry what$",
            r"^one more time$",
        ),
    ),
    ("stop", _rx(r"^(?:stop|stop talking|stop it|shut up|be quiet|quiet|enough|that is enough|silence|hush)$")),
    (
        "cancel",
        _rx(
            r"^(?:cancel|cancel that|cancel it|never ?mind|forget it|forget that)$",
            r"^(?:stop|cancel) (?:watching|the watch|waiting)$",
        ),
    ),
]

# "I'm going to <place>"; the place must not start with a verb ("I'm going to turn left") or be about lanes.
_DESTINATION = _rx(
    r"^(?:i am |we are |i'm )?(?:going|heading|headed|driving) (?:over )?to (?P<p1>.+)$",
    r"^(?:take (?:me|us)|navigate|directions|drive|route (?:me|us)|get (?:me|us)|let us go|go|bring (?:me|us))"
    r" to (?P<p2>.+)$",
    r"^(?:set |change )?(?:the |my )?(?:destination|address)(?: to| is)? (?P<p3>.+)$",
    r"^(?:new destination|find (?:me )?(?:a route|directions) to)(?: is)? (?P<p4>.+)$",
)
_NOT_A_PLACE = _rx(
    r"^(?:turn|go|park|stop|merge|change|switch|pass|make|take|need|be|try|wait|get|move|pull|back|run|hit|miss|"
    r"drive|stay|keep|follow|exit|use|cross|head|do|have|see|let|the (?:left|right|middle|next|first|second|third)"
    r"|that (?:lane|exit)|this (?:lane|exit)|left|right|straight)\b",
    r"\blane\b",
)

_GO_WATCH = _rx(
    r"\b(?:tell|let|alert|notify|warn|ping|remind|show) (?:me|us)\b.*\b(?:my turn|our turn|i can go|we can go|"
    r"i am up|i should go|to go|safe to go|go time)\b",
    r"^(?:say|shout|speak up) when (?:i can go|it is my turn|my turn)\b",
)
_LIGHT_WATCH = _rx(
    r"\b(?:tell|let|alert|notify|warn|ping|remind) (?:me|us)\b.*\b(?:green|light|signal|changes?|turns?)\b",
    r"^(?:watch|wait for|keep an eye on|monitor) (?:the |that |this )?(?:light|signal)\b",
    r"^(?:say|shout|speak up) when\b.*\b(?:green|light|changes?)\b",
)

_GAP = _rx(
    r"\b(?:can|could|should|may|shall) (?:i|we) (?:merge|pass|overtake|change lanes?|switch lanes?|cut in|squeeze in|"
    r"pull out|pull in|make (?:it|the light|that light|this light|the yellow|the turn)|beat (?:the|that) light|"
    r"turn now|go around|get over now|move over now|go for it|gun it|run (?:the|this|that) (?:light|yellow))\b",
    r"\bis (?:it|this|now) (?:a )?(?:safe|clear|ok|okay|good|a good time) (?:time )?to "
    r"(?:merge|pass|overtake|change lanes?|switch lanes?|pull out|turn|go|cross|move over|get over)\b",
    r"\b(?:is|are) (?:the )?(?:lane|left lane|right lane|left|right|coast|road|way|traffic|my blind spot|my left|"
    r"my right) (?:clear|free|open|empty)\b",
    r"\bblind ?spot\b",
    r"\bis (?:it|the coast) clear\b",
    r"\b(?:is there|are there|is anyone|is anybody|is someone|is somebody) (?:a car |a vehicle |cars |anyone |"
    r"anybody |someone |somebody |anything )?(?:next to|beside|behind|alongside|on my (?:left|right) side)\b",
    r"\b(?:enough|big enough) (?:room|space|gap)\b",
    r"\b(?:gap|opening) (?:in|between) (?:the )?(?:traffic|cars)\b",
)

_RIGHT_OF_WAY: list[tuple[str, re.Pattern[str]]] = [
    (
        "position",
        _rx(
            r"\bhow many (?:cars |vehicles |people |drivers )?(?:are |is )?"
            r"(?:ahead of|before|in front of|waiting before|going before|first|ahead)\b",
            r"\bwhat (?:number|position|place) am i\b",
            r"\bwhere am i in (?:the )?(?:line|queue|order)\b",
            r"\bwhich (?:number|position|place) am i\b",
        ),
    ),
    (
        "can_i_go",
        _rx(
            r"^(?:can|may|should|do|could) (?:i|we) go(?: now| yet| first| ahead| already| next)?$",
            r"^(?:can|may|should|do) (?:i|we) go(?: now)? or (?:not|wait|should i wait)$",
            r"\b(?:is it|it is|is this) (?:my|our) turn\b",
            r"\bam i (?:next|first|up|good to go|clear to go)\b",
            r"^when (?:can|do|should|will) (?:i|we) (?:get to )?go\b",
            r"\bdo (?:i|we) have (?:the )?right of way\b",
            r"^(?:is it )?(?:my|our) turn(?: yet| now)?$",
            r"^(?:can|should) (?:i|we) (?:pull up|move up|proceed|drive)(?: now)?$",
            r"\bhow (?:much )?longer (?:do|should|until) (?:i|we)\b",
            r"^(?:should i|do i) (?:wait|keep waiting)\b",
        ),
    ),
    (
        "who_first",
        _rx(
            r"\bwho (?:is |was |got |came |arrived |stopped |goes |went |should go |has to go |gets to go )?"
            r"(?:here |there )?(?:first|next|before me|ahead of me)\b",
            r"\bwhose turn\b",
            r"\bwho has (?:the )?right of way\b",
            r"\b(?:what|which) (?:is the )?order\b",
            r"\bwho (?:goes|should go|has to go)\b",
            r"\bwho (?:is|was) (?:waiting|stopped|here)\b",
            r"\bin what order\b",
        ),
    ),
]

_NAV: list[tuple[str, re.Pattern[str]]] = [
    ("cancel_route", _rx(r"\b(?:cancel|stop|end|clear|quit) (?:the |my )?(?:navigation|route|directions|nav|trip)\b")),
    (
        "missed_turn",
        _rx(
            r"\b(?:did|have) (?:i|we) (?:miss|missed|passed?|go past|gone past|drive past|driven past)"
            r" (?:my|the|our) (?:turn|exit|street|destination)\b",
            r"\b(?:i|we) (?:missed|passed) (?:my|the|our) (?:turn|exit|street)\b",
            r"\bam i (?:still )?(?:on (?:the )?(?:route|right road|right track)|lost)\b",
            r"\b(?:reroute|re route|new route)\b",
        ),
    ),
    (
        "exit",
        _rx(
            r"\b(?:which|what) exit\b",
            r"\bis (?:my exit|it|this) (?:this one|the next|this exit|my exit)\b",
            r"\bmy exit\b",
            r"\bthis exit or\b",
            r"\bexit number\b",
        ),
    ),
    (
        "next_step",
        _rx(
            r"\b(?:next|my next|the next) (?:turn|step|instruction|direction|maneuver)\b",
            r"\bwhere do (?:i|we) (?:turn|go next|go from here)\b",
            r"\bwhat (?:is |comes |happens )?(?:after|next after) (?:this|that|the)\b",
            r"\bwhich way (?:do|should) (?:i|we)\b",
            r"\bwhere (?:to )?next\b",
            r"\b(?:do|should) (?:i|we) turn (?:here|at this|at the next|left or right)\b",
            r"\bwhat street (?:do|should) (?:i|we) turn\b",
        ),
    ),
    (
        "eta",
        _rx(
            r"\bhow (?:far|long|much longer|many (?:miles|minutes))\b.*\b(?:turn|exit|destination|there|arrive|"
            r"left|go|drive|trip)\b",
            r"\beta\b",
            r"\bwhen (?:will|do|should) (?:we|i) (?:get|arrive|be there)\b",
            r"\b(?:are we|almost) there\b",
            r"\barrival time\b",
            r"\btime (?:left|remaining)\b",
            r"\bhow (?:far|long) (?:away|to go)\b",
        ),
    ),
    (
        "destination_info",
        _rx(r"\bwhere am i going\b", r"\bwhat is (?:my|the|our) destination\b", r"\bwhere are we going\b"),
    ),
]

_LANE = _rx(
    r"\b(?:which|what) lane\b",
    r"\bam i in the (?:right|correct|wrong|proper) lane\b",
    r"\bis (?:this|it|that|my lane) (?:the )?(?:right|correct|wrong|proper) lane\b",  # not "is the right lane closed"
    r"\blanes? for\b",
    r"\bshould (?:i|we) (?:move|get|be|go|change lanes|switch lanes) (?:over |in |into )?(?:to )?(?:the )?"
    r"(?:left|right|over|middle|center|lane)\b",
    r"\b(?:move|change|switch|get) (?:over|lanes?) (?:for|to get to|to take|to reach)\b",
    r"\bwhich lanes?\b",
    r"\bstay in (?:this|my) lane\b",
)


def route(text: str) -> Route:
    """Route one transcribed question (see the module docstring for the rule order)."""
    norm = normalize(text)
    question = strip_wake_word(text)

    def make(kind: RouteKind, intent: str, place: str | None = None) -> Route:
        return Route(kind, intent, norm, question, place)

    if not norm:
        return make(RouteKind.EMPTY, "empty")

    for intent, pattern in _CONTROLS:
        if pattern.search(norm):
            return make(RouteKind.CONTROL, intent)

    match = _DESTINATION.search(norm)
    if match is not None:
        place = next(v for v in match.groupdict().values() if v).strip()
        place = re.sub(r"\s+(?:now|then|thanks|thank you)$", "", place)
        if place and not _NOT_A_PLACE.search(place):
            return make(RouteKind.DESTINATION, "set_destination", place)

    if _GO_WATCH.search(norm):
        return make(RouteKind.WATCH, "go_watch")
    if _LIGHT_WATCH.search(norm):
        return make(RouteKind.WATCH, "light_watch")

    for intent, pattern in _NAV[:2]:  # "did I miss my turn" before right-of-way's "my turn"
        if pattern.search(norm):
            return make(RouteKind.NAV, intent)

    if _GAP.search(norm):
        return make(RouteKind.GAP_CHECK, "gap_check")

    for intent, pattern in _RIGHT_OF_WAY:
        if pattern.search(norm):
            return make(RouteKind.RIGHT_OF_WAY, intent)

    if _LANE.search(norm):
        return make(RouteKind.LANE, "lane")

    for intent, pattern in _NAV[2:]:
        if pattern.search(norm):
            return make(RouteKind.NAV, intent)

    return make(RouteKind.MODEL, "scene")
