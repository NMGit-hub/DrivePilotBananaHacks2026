"""Question router: decides which handler answers a transcribed question (PLAN.md "How questions are routed").

A fast keyword/regex router, so routing costs microseconds and local answers stay instant:

- **control**: quiet mode, normal mode, repeat that, stop, never mind (voice controls);
- **destination**: "I'm going to <place>" (navigation by voice);
- **watch**: "tell me when it's green" (light) and "tell me when I can go" (turn at a stop, or the light);
- **gap check**: split-second judgments ("can I merge now?", "is my lane clear?", "will I make the light?",
  "tell me when it's safe to merge"): refused while moving; never answered with a "yes";
- **right of way** (local, instant): "who got here first?", "can I go?", "should I wait?", "how many cars are
  ahead of me?";
- **nav**: route questions ("did I miss my turn?", "which exit?", "how far to my turn?", "where do I go?");
- **lane**: "which lane for 520 east?", "am I in the right lane?", "should I change lanes for my exit?";
- **model**: everything else (signs, parking, lights, road awareness) goes to the vision model.

Rules are checked in a fixed order and the first match wins; the order encodes the tricky cases, e.g. "can I go
right on red?" is a sign question (model), "did I miss my turn?" is navigation while "is it my turn?" is right
of way, and "should I change lanes for my exit?" is lane guidance while "can I change lanes?" is a gap check.
Split-second questions are matched by their shape (a question frame + a maneuver) rather than fixed sentences,
because missing one means answering it while moving. The router is pure (text in, `Route` out) and is tested
with two tables: the planned phrasings and an adversarial set from the M3 review.
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


# ---------------------------------------------------------------------------------------- normalizing

_WAKE_NAME = r"(?:drive|dr\.?|dry|driving|drove)[\s-]*(?:pilot|pylot|pilots|pylon|pile it|pilate)"
_WAKE_LEAD = re.compile(rf"^\s*(?:(?:hey|hi|okay|ok|a|yo|hay)[\s,]+)?{_WAKE_NAME}\b[\s,.!?:;-]*", re.IGNORECASE)
_WAKE_TAIL = re.compile(rf"[\s,]+(?:hey\s+)?{_WAKE_NAME}[\s.!?]*$", re.IGNORECASE)
_CONTRACTIONS = {
    "what's": "what is",
    "whats": "what is",
    "who's": "who is",
    "whos": "who is",
    "it's": "it is",
    "its": "it is",
    "where's": "where is",
    "wheres": "where is",
    "how's": "how is",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
    "theres": "there is",
    "here's": "here is",
    "i'm": "i am",
    "im": "i am",
    "let's": "let us",
    "lets": "let us",
    "can't": "can not",
    "cant": "can not",
    "cannot": "can not",
    "don't": "do not",
    "dont": "do not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "didn't": "did not",
    "didnt": "did not",
    "i've": "i have",
    "i'll": "i will",
    "we're": "we are",
    "you're": "you are",
    "when's": "when is",
    "whens": "when is",
    "gonna": "going to",
    "wanna": "want to",
}
_CONTRACTION = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b")
_FILLER = re.compile(r"\b(?:please|um+|uh+|erm|hmm+|just|actually|hey|like|real quick)\b")
# Leading words that carry no meaning ("okay, so can I go?"); "ok to merge" / "okay if I go" keep theirs.
_LEADING_FILLER = re.compile(r"^(?:so|okay|ok|well|and|alright|all right|yeah|yes|now|oh)\b(?! (?:to|if|for)\b)\s*")


def strip_wake_word(text: str) -> str:
    """The question without a leading or trailing wake word ("Hey DrivePilot, ..." / "..., DrivePilot")."""
    text = text.replace("’", "'").replace("‘", "'").strip()
    return _WAKE_TAIL.sub("", _WAKE_LEAD.sub("", text)).strip()


def normalize(text: str) -> str:
    """Lowercase, drop the wake word and filler words, expand common contractions, strip punctuation."""
    text = strip_wake_word(text).lower()
    text = _CONTRACTION.sub(lambda m: _CONTRACTIONS[m.group(1)], text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = _FILLER.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" '")
    previous = None
    while previous != text:  # "okay so um can i go" -> "can i go"
        previous, text = text, _LEADING_FILLER.sub("", text).strip()
    return text


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# ---------------------------------------------------------------------------------------- controls

_CONTROLS: list[tuple[str, re.Pattern[str]]] = [
    (
        "normal",
        _rx(
            r"^(?:turn on |switch to |go to |go back to |back to |use )?(?:normal|regular|loud) mode(?: on| now)?$",
            r"^(?:turn off|exit|disable|leave|stop|end|cancel|no more) (?:the )?(?:quiet|silent) mode$",
            r"^(?:quiet|silent) mode off$",
            r"^(?:unmute|turn on|enable) (?:the )?alerts$",
            r"^alerts on$",
            r"^back to normal$",
        ),
    ),
    (
        "quiet",
        _rx(
            r"^(?:turn on |switch to |go to |go into |enter |use |enable |activate )?(?:quiet|silent) mode"
            r"(?: on| now)?$",
            r"^(?:mute|silence|turn off|disable|stop) (?:the |all )?alerts$",
            r"^go quiet$",
        ),
    ),
    (
        "repeat",
        _rx(
            r"^(?:can|could|would) you (?:repeat|say) (?:that|it)(?: again)?$",
            r"^(?:repeat|say) (?:that|it|again|that again|it again|the last thing)(?: again)?$",
            r"^repeat$",
            r"^what did you (?:just )?say$",
            r"^come again$",
            r"^pardon(?: me)?$",
            r"^sorry(?: what)?$",
            r"^what$",
            r"^huh$",
            r"^what was that$",
            r"^one more time$",
        ),
    ),
    (
        "stop",
        _rx(
            r"^(?:no |ok |okay )?(?:stop|stop talking|stop it|stop that|stop now|shut up|be quiet|quiet|enough|"
            r"that is enough|silence|hush|shush|shh+)(?: (?:stop|now|please))*$",
        ),
    ),
    (
        "cancel",
        _rx(
            r"^(?:cancel|cancel that|cancel it|never ?mind|forget it|forget that|do not tell me|no need)$",
            r"^(?:stop|cancel) (?:watching|the watch|waiting)(?: (?:the|that) (?:light|signal))?$",
            r"^cancel (?:the )?(?:alert|alerts|reminder|watch)$",
        ),
    ),
]

# ----------------------------------------------------------------------------------------- navigation

_NAV_FIRST: list[tuple[str, re.Pattern[str]]] = [
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
        "next_step",
        _rx(
            r"\b(?:tell|show|let) (?:me|us)(?: know)? (?:when|where|which way) to (?:go|turn|exit)\b",
            r"\b(?:tell|show|let) (?:me|us)(?: know)? (?:where|which way) (?:i|we) (?:go|turn|should go)\b",
            r"\b(?:tell|show|let) (?:me|us)(?: know)? (?:about )?(?:my|the|our) next (?:turn|step|exit)\b",
            r"\b(?:tell|let) (?:me|us)(?: know)? when (?:my|our|the) (?:turn|exit) is coming(?: up)?\b",
            r"\b(?:tell|let) (?:me|us)(?: know)? if (?:i|we) need to (?:go|turn|exit|keep)\b",
        ),
    ),
]

_NAV_LATE: list[tuple[str, re.Pattern[str]]] = [
    (
        "exit",
        _rx(
            r"\b(?:which|what) exit\b",
            r"\bis (?:my exit|it|this) (?:this one|the next(?: one| exit)?$|this exit|my exit)",
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
            r"\bis (?:it|my turn|the turn) (?:at )?the next (?:light|street|intersection|corner|exit|left|right)\b",
        ),
    ),
    (
        "eta",
        _rx(
            r"\bhow (?:far|long) (?:to|until|till|is it to|away is) (?:my|the|our) (?:turn|exit|destination|"
            r"next turn|next exit)\b",
            r"\bhow (?:much )?(?:longer|farther|further)\b(?! (?:has|have|is|are|was|were|did)\b)",
            r"\bhow (?:far|long) (?:until|till|to|before) (?:we|i) (?:get there|arrive|get home|are there)\b",
            r"\bhow (?:far|long) (?:to go|away|left)$",
            r"\bhow (?:far|long) is (?:the|my|our) (?:drive|trip)\b",
            r"\beta\b",
            r"\bwhen (?:will|do|should) (?:we|i) (?:get|arrive|be there)\b",
            r"\b(?:are we|almost) there\b",
            r"\barrival time\b",
            r"\btime (?:left|remaining)\b",
        ),
    ),
    (
        "destination_info",
        _rx(r"\bwhere am i going\b", r"\bwhat is (?:my|the|our) destination\b", r"\bwhere are we going\b"),
    ),
]

# --------------------------------------------------------------------------------------- destination

# "I'm going to <place>"; not "I'm going to turn left / speed up / the exit".
_DESTINATION = _rx(
    r"^(?:i am |we are )?(?:going|heading|headed|driving) (?:over )?to (?P<p1>.+)$",
    r"^(?:take (?:me|us)|navigate|directions|drive|route (?:me|us)|get (?:me|us)|let us go|go|bring (?:me|us))"
    r" to (?P<p2>.+)$",
    r"^(?:set |change )?(?:the |my )?(?:destination|address)(?: to| is)? (?P<p3>.+)$",
    r"^(?:new destination|find (?:me )?(?:a route|directions) to)(?: is)? (?P<p4>.+)$",
    r"^(?:take (?:me|us)|navigate|go|drive|get (?:me|us)|head) (?P<p5>home)$",
)
_VERBS = (
    r"turn|go|park|stop|merge|change|switch|pass|make|take|need|be|try|wait|get|move|pull|back|run|hit|miss|"
    r"drive|stay|keep|follow|exit|use|cross|head|do|have|see|let|speed|slow|cut|overtake|accelerate|brake|honk|"
    r"reverse|yield|squeeze|beat|stop at|catch|wait for"
)
_NOT_A_PLACE = _rx(
    rf"^(?:{_VERBS})(?:$| (?:left|right|here|there|now|up|down|over|in|out|around|straight|back|off|on|it|this|"
    r"that|them|him|her|the|a|an|to|for|into|onto|through|across|behind|ahead|past|lanes?|faster|slower|"
    r"carefully|somewhere)\b)",
    r"^(?:the |my |this |that |next )?(?:exit|stop line|light|intersection|corner|left|right|shoulder|next exit|"
    r"lane|left lane|right lane|middle lane|other lane|crosswalk|curb|gas pedal|brakes?)$",
    r"\b(?:left|right|middle|next|other|that|this|turn|passing|fast|slow) lane\b",
)

# ------------------------------------------------------------------------------------------- watches

_GAP_WATCH = _rx(
    r"\b(?:tell|let|alert|notify|warn|ping|remind) (?:me|us)(?: know)? (?:when|once|as soon as|if) (?:it is |the "
    r"(?:\w+ )?(?:lane|road|way|traffic|coast) is |traffic is |everything is )?(?:safe|clear|open)\b",
    r"\b(?:tell|let|alert|notify) (?:me|us)(?: know)? (?:when|once|as soon as) (?:i|we) can (?:merge|turn|pull out|"
    r"change lanes?|switch lanes?|get over|move over|pass|cut in|go left|go right|cross)\b",
)
_GO_WATCH = _rx(
    r"\b(?:tell|let|alert|notify|warn|ping|remind) (?:me|us)(?: know)? (?:when|once|as soon as) (?:i|we) can go\b",
    r"\b(?:tell|let|alert|notify) (?:me|us)(?: know)? (?:when|once|as soon as) (?:it is|it is my|it is our) "
    r"(?:my |our )?turn\b",
    r"\b(?:tell|let|alert|notify) (?:me|us)(?: know)? when (?:my|our) turn\b(?! is coming)",
    r"\b(?:tell|let) (?:me|us)(?: know)? when to go\b",
    r"\b(?:tell|let) (?:me|us)(?: know)? when (?:the|that) (?:car|truck|bus|van) (?:ahead |in front )?"
    r"(?:goes|moves|leaves)\b",
    r"^(?:say|shout|speak up) when (?:i can go|it is my turn|my turn)\b",
)
_LIGHT_WATCH = _rx(
    r"\b(?:tell|let|alert|notify|warn|ping|remind) (?:me|us)(?: know)? (?:when|once|as soon as|if) "
    r"(?:the |that |this )?(?:light|signal|it) (?:is|turns|goes|changes|gets|will be|turn|go|change)(?: to)?"
    r"(?: green)?\b",
    r"\b(?:tell|let|alert|notify) (?:me|us)(?: know)? (?:when|once|as soon as) (?:it is|it turns|it goes) green\b",
    r"\b(?:tell|let) (?:me|us)(?: know)? when (?:the |that )?(?:light|signal) (?:is green|turns|changes|goes)\b",
    r"^(?:can you |could you |please )?(?:watch|wait for|keep an eye on|monitor) (?:the |that |this )?"
    r"(?:light|signal)\b",
    r"^(?:say|shout|speak up) when\b.*\b(?:green|light|changes?)\b",
)

# ---------------------------------------------------------------------------------------- gap checks

_ACTION = (
    r"(?:merge|pass|overtake|get over|move over|cut in|squeeze in|pull out|pull in|go around|change lanes?|"
    r"switch lanes?|(?:get|move|change|switch|go) (?:in|into|to|over to|over into)(?: the)?(?: \w+)? lane|"
    r"turn(?: left| right)? now|go (?:left|right) now|cross now|"
    r"make (?:it|the light|that light|this light|the yellow|the turn|this turn|this left|this right|the left|"
    r"the right|that turn|that left)|beat (?:the|that|this) (?:light|car|truck|bus|yellow)|"
    r"go through (?:the|this|that) (?:yellow|light)|run (?:the|this|that) (?:light|yellow|red)|"
    r"go before (?:that|the|this|him|her|them)|get on (?:the )?(?:highway|freeway|interstate|on ramp)|brake|"
    r"stop or go|get in|slip in|jump in)"
)
_GO = r"(?:go|turn|cross|proceed|drive|pull out|go now|turn now)"
_GAP = _rx(
    rf"\b(?:can|could|should|may|shall|will|would) (?:i|we)(?: still| just| safely)? {_ACTION}\b",
    rf"\b(?:am i|are we) going to {_ACTION}\b",
    rf"\bi am going to (?:try to )?{_ACTION}\b",
    rf"\bdo (?:i|we) have (?:enough )?(?:time|room|space) to {_ACTION}\b",
    rf"\b(?:is there|there is|do i have|do we have) (?:enough )?(?:room|space|time|a gap) (?:to|for me to) {_ACTION}\b",
    rf"\b(?:is it |it is |am i |are we |is that )?(?:safe|ok|okay|clear|good|alright) (?:for (?:me|us) )?to "
    rf"(?:{_ACTION}|{_GO})\b",
    rf"\bis it (?:ok|okay|alright|safe|fine) if (?:i|we) (?:{_ACTION}|{_GO})\b",
    r"\b(?:am i|are we) (?:clear|safe|good|ok|okay)(?: to \w+)?$",
    r"^(?:is it |is that )?(?:safe|clear)(?: now| yet| enough)?$",
    r"^(?:merge|go|turn|pull out|cross) now$",
    r"^(?:should|do) (?:i|we) (?:stop|brake|slow down)(?: or (?:go|keep going))?$",
    r"\bstop or (?:go|keep going)\b",
    r"\bwill (?:i|we) make (?:it|the light|the yellow|that light)\b",
    r"\b(?:is|are) (?:it|the coast|everything|(?:the |my )?(?:\w+ ){0,3}?(?:lane|left|right|side|road|way|traffic|"
    r"intersection|crosswalk|blind spot|highway|on ramp))"
    r"(?: (?:to|on|next to) (?:my|the|our|me)(?: (?:left|right|side))?)?(?: ahead)? (?:clear|free|open|empty)\b",
    r"^(?:the |my )?(?:left |right |middle |next |other )?(?:lane|side)(?: to my (?:left|right))? "
    r"(?:clear|free|open|empty)$",
    r"\b(?:is|are) (?:there )?(?:anyone|anybody|someone|somebody|a car|cars|anything|a vehicle|a bike|a truck) "
    r"(?:in|on|to|next to|beside|behind|alongside) (?:my|me)\b",
    r"^(?:anyone|anybody|someone|somebody|anything) (?:next to|beside|behind|on my|in my|to my)\b",
    r"\bblind ?spot\b",
    r"\b(?:is|will|does) (?:that|the|this) (?:car|truck|bus|guy|driver|van|suv|cyclist)(?: \w+)?"
    r" (?:going to |about to )?(?:stop|let me in|let us in|yield|wait)\b",
    r"\b(?:is|will) (?:he|she|they) (?:going to |about to )?(?:stop|let me in|let us in|yield|wait)\b",
    r"\b(?:enough|big enough) (?:room|space|gap)\b",
    r"\b(?:gap|opening) (?:in|between) (?:the )?(?:traffic|cars)\b",
)

# --------------------------------------------------------------------------------------- right of way

_RIGHT_OF_WAY: list[tuple[str, re.Pattern[str]]] = [
    (
        "position",
        _rx(
            r"\bhow many (?:cars |vehicles |people |drivers )?(?:are |is )?"
            r"(?:ahead of|before|in front of|waiting before|going before|first|ahead)\b",
            r"\bhow many (?:cars |vehicles )?(?:go|goes|have to go) (?:before|ahead of) (?:me|us)\b",
            r"\b(?:what|which) (?:number|position|place) am i\b",
            r"\bwhere am i in (?:the )?(?:line|queue|order)\b",
        ),
    ),
    (
        "should_wait",
        _rx(r"^(?:should|do|must) (?:i|we) (?:wait|keep waiting|still wait|stay put|hold)\b"),
    ),
    (
        "can_i_go",
        _rx(
            r"^(?:can|may|should|could|do|shall) (?:i|we) (?:still )?go(?:$| (?:now|yet|first|ahead|already|next|"
            r"here|then|now then|yet or not|or not|or what|now or not|now that\b.*|since\b.*))",
            r"^(?:can|may|should|do) (?:i|we) go(?: now)? or (?:not|wait|should i wait)$",
            r"\b(?:is it|it is|is this) (?:my|our) turn\b",
            r"^(?:it is |is it )?(?:my|our) turn(?: yet| now)?$",
            r"\bis it (?:me|us) next\b",
            r"\bam i (?:next|first|up|good to go)\b(?! to\b| against\b| on\b| in\b| at\b| by\b)",
            r"\bwhen (?:is|will be|will it be) (?:my|our) turn\b",
            r"^when (?:can|do|should|will) (?:i|we) (?:get to )?go\b",
            r"\bdo (?:i|we) have (?:the )?right of way\b",
            r"^(?:can|should) (?:i|we) (?:pull up|move up|proceed|drive)(?: now)?$",
        ),
    ),
    (
        "who_first",
        _rx(
            r"\bwho (?:is |was |got |came |arrived |stopped |goes |went |should go |has to go |gets to go )?"
            r"(?:here |there )?(?:first|next|before me|ahead of me)\b(?! to\b)",
            r"\bwhose turn\b",
            r"\bwho has (?:the )?right of way\b",
            r"\b(?:what|which) (?:is the )?order\b",
            r"\bwho (?:goes|should go|has to go)\b",
            r"\bwho (?:is|was) (?:waiting|stopped|here)\b",
            r"\bin what order\b",
            r"\bwhich (?:car|truck|vehicle|one) (?:goes|got here|went|arrived|stopped|was here|is) first\b",
            r"\b(?:was|am) (?:i|we) (?:here )?first\b",
            r"\bdid (?:i|we) (?:get here|arrive|stop|get there) first\b",
            r"\b(?:does|did) the (?:\w+ )?(?:car|truck|bus|van|suv|pickup|guy) (?:go|get) "
            r"(?:first|before me|before us)\b",
            r"\bdo (?:i|we) go (?:before|after) (?:the|that|this)\b",
        ),
    ),
]

# ------------------------------------------------------------------------------------------------ lanes

_LANE_FIRST = _rx(
    r"\b(?:change|switch|move|get) (?:over|lanes?|into the \w+ lane)(?: \w+)? (?:for|to get to|to take|to reach)\b",
    r"\blanes? for\b",
)
_LANE = _rx(
    r"\b(?:which|what) lane\b",
    r"\bam i in the (?:right|correct|wrong|proper) lane\b",
    r"\bis (?:this|it|that|my lane) (?:the )?(?:right|correct|wrong|proper) lane\b",  # not "is the right lane closed"
    r"\bshould (?:i|we) (?:move|get|be|go) (?:over |in |into )?(?:to )?(?:the )?(?:left|right|over|middle|center)\b",
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

    for intent, pattern in _NAV_FIRST:  # "did I miss my turn", "tell me where to go" before right of way / watches
        if pattern.search(norm):
            return make(RouteKind.NAV, intent)

    if _GAP_WATCH.search(norm):  # "tell me when it's safe to merge": a gap judgment, never a watch
        return make(RouteKind.GAP_CHECK, "gap_watch")

    match = _DESTINATION.search(norm)
    if match is not None:
        place = next(v for v in match.groupdict().values() if v).strip()
        place = re.split(r"\s+(?:how|what|when|is|and then|then|thanks|thank you)\b", place)[0].strip()
        place = re.sub(r"\s+(?:now|please)$", "", place)
        if place and not _NOT_A_PLACE.search(place) and not _GAP.search(norm):
            return make(RouteKind.DESTINATION, "set_destination", place)

    if _GO_WATCH.search(norm):
        return make(RouteKind.WATCH, "go_watch")
    if _LIGHT_WATCH.search(norm):
        return make(RouteKind.WATCH, "light_watch")

    if _LANE_FIRST.search(norm):  # "should I change lanes for my exit?" is guidance, not a gap check
        return make(RouteKind.LANE, "lane")

    if _GAP.search(norm):
        return make(RouteKind.GAP_CHECK, "gap_check")

    for intent, pattern in _RIGHT_OF_WAY:
        if pattern.search(norm):
            return make(RouteKind.RIGHT_OF_WAY, intent)

    if _LANE.search(norm):
        return make(RouteKind.LANE, "lane")

    for intent, pattern in _NAV_LATE:
        if pattern.search(norm):
            return make(RouteKind.NAV, intent)

    return make(RouteKind.MODEL, "scene")
