"""Router table: phrasings as Whisper transcribes them → handler (BUILD_PLAN M3: >= 60 phrasings)."""

from __future__ import annotations

import pytest

from voice.router import RouteKind, normalize, route, strip_wake_word

C, D, W, G, R, N, L, M = (
    RouteKind.CONTROL,
    RouteKind.DESTINATION,
    RouteKind.WATCH,
    RouteKind.GAP_CHECK,
    RouteKind.RIGHT_OF_WAY,
    RouteKind.NAV,
    RouteKind.LANE,
    RouteKind.MODEL,
)

TABLE: list[tuple[str, RouteKind, str]] = [
    # --- voice controls (PLAN.md "Voice controls")
    ("Quiet mode.", C, "quiet"),
    ("Hey DrivePilot, quiet mode", C, "quiet"),
    ("Switch to quiet mode please", C, "quiet"),
    ("Mute the alerts.", C, "quiet"),
    ("Normal mode.", C, "normal"),
    ("Go back to normal mode", C, "normal"),
    ("Quiet mode off.", C, "normal"),
    ("Repeat that.", C, "repeat"),
    ("Can you repeat that?", M, "scene"),  # deliberate: only the short commands are controls...
    ("Say that again.", C, "repeat"),
    ("What did you just say?", C, "repeat"),
    ("Stop.", C, "stop"),
    ("Stop talking.", C, "stop"),
    ("Be quiet!", C, "stop"),
    ("Never mind.", C, "cancel"),
    ("Cancel that.", C, "cancel"),
    # --- destination by voice
    ("I'm going to Redmond Town Center.", D, "set_destination"),
    ("Hey Drive Pilot, I'm going to the Space Needle", D, "set_destination"),
    ("Take me to Seattle Children's Hospital.", D, "set_destination"),
    ("Navigate to 1600 Amphitheatre Parkway", D, "set_destination"),
    ("Let's go to Costco.", D, "set_destination"),
    ("Set destination to Bellevue Square.", D, "set_destination"),
    ("We're heading to the airport", D, "set_destination"),
    ("I'm going to turn left here, is that allowed?", M, "scene"),  # "going to" + verb is not a place
    ("I'm going to park here, is that okay?", M, "scene"),
    # --- watches
    ("Tell me when it's green.", W, "light_watch"),
    ("Tell me when the light turns green", W, "light_watch"),
    ("Let me know when the light changes.", W, "light_watch"),
    ("Hey DrivePilot, alert me when it goes green", W, "light_watch"),
    ("Watch the light for me.", W, "light_watch"),
    ("Say when it's green.", W, "light_watch"),
    ("Tell me when it's my turn.", W, "go_watch"),
    ("Let me know when I can go.", W, "go_watch"),
    # --- split-second judgments (refused while moving)
    ("Can I merge now?", G, "gap_check"),
    ("Is it safe to change lanes?", G, "gap_check"),
    ("Can I change lanes?", G, "gap_check"),
    ("Is my blind spot clear?", G, "gap_check"),
    ("Is anyone in my blind spot?", G, "gap_check"),
    ("Can I pass this truck?", G, "gap_check"),
    ("Can I make the light?", G, "gap_check"),
    ("Is the left lane clear?", G, "gap_check"),
    ("Is there a car next to me?", G, "gap_check"),
    ("Is it clear?", G, "gap_check"),
    ("Is it safe to pull out?", G, "gap_check"),
    ("Is it safe to go?", G, "gap_check"),  # handled by context: right-of-way answer at a stop
    ("Is there enough room to merge?", G, "gap_check"),
    # --- right of way (local)
    ("Who got here first?", R, "who_first"),
    ("Hey DrivePilot, who got here first?", R, "who_first"),
    ("Who was first?", R, "who_first"),
    ("Who's first?", R, "who_first"),
    ("Who goes first?", R, "who_first"),
    ("Who's next?", R, "who_first"),
    ("Whose turn is it?", R, "who_first"),
    ("Who has the right of way?", R, "who_first"),
    ("What's the order here?", R, "who_first"),
    ("Can I go?", R, "can_i_go"),
    ("Can I go now?", R, "can_i_go"),
    ("Should I go?", R, "can_i_go"),
    ("When can I go?", R, "can_i_go"),
    ("Is it my turn?", R, "can_i_go"),
    ("Is it my turn yet?", R, "can_i_go"),
    ("My turn?", R, "can_i_go"),
    ("Am I next?", R, "can_i_go"),
    ("Do I have the right of way?", R, "can_i_go"),
    ("Should I wait?", R, "can_i_go"),
    ("How many cars are ahead of me?", R, "position"),
    ("What number am I?", R, "position"),
    ("Where am I in the line?", R, "position"),
    # --- tricky: look like right of way but aren't
    ("Can I go right on red here?", M, "scene"),  # a sign question
    ("Can I turn right on red?", M, "scene"),
    ("Can I go left here?", M, "scene"),
    ("Can I go straight from this lane?", M, "scene"),
    ("Can I go faster here?", M, "scene"),
    ("Did I miss my turn?", N, "missed_turn"),  # navigation, not "my turn" at a stop
    ("Did we miss the exit?", N, "missed_turn"),
    ("Is the light green?", M, "scene"),  # light questions go to the model (arrows, glare)
    ("Is the left arrow green?", M, "scene"),
    ("Is that a stop sign?", M, "scene"),
    # --- navigation (route questions)
    ("Is my exit this one or the next?", N, "exit"),
    ("Which exit do I take?", N, "exit"),
    ("Which exit?", N, "exit"),
    ("What's after this light?", N, "next_step"),
    ("What's my next turn?", N, "next_step"),
    ("Where do I turn?", N, "next_step"),
    ("How far to my turn?", N, "eta"),
    ("How long until we get there?", N, "eta"),
    ("When will we arrive?", N, "eta"),
    ("Are we there yet?", N, "eta"),
    ("Where am I going?", N, "destination_info"),
    ("Cancel the navigation.", N, "cancel_route"),
    ("Stop navigation.", N, "cancel_route"),
    ("Am I still on the route?", N, "missed_turn"),
    # --- lanes
    ("Which lane for 520 east?", L, "lane"),
    ("Am I in the right lane?", L, "lane"),
    ("Am I in the correct lane?", L, "lane"),
    ("What lane should I be in?", L, "lane"),
    ("Is this the right lane for the exit?", L, "lane"),
    ("Should I move over to the left?", L, "lane"),
    # --- everything else: the vision model
    ("What's the speed limit?", M, "scene"),
    ("What did that sign say?", M, "scene"),
    ("Can I park here?", M, "scene"),
    ("What does the red curb mean?", M, "scene"),
    ("Is this spot EV only?", M, "scene"),
    ("Is the road closed ahead?", M, "scene"),
    ("Why is that car stopped?", M, "scene"),
    ("What does that construction sign want?", M, "scene"),
    ("Can I use the express lane?", M, "scene"),
    ("Is the right lane closed?", M, "scene"),
    ("How many lanes are there?", M, "scene"),
    ("What color is the car on my left?", M, "scene"),
    ("Is that person crossing?", M, "scene"),
]


@pytest.mark.parametrize(("text", "kind", "intent"), TABLE, ids=[t for t, _, _ in TABLE])
def test_router_table(text: str, kind: RouteKind, intent: str) -> None:
    result = route(text)
    assert (result.kind, result.intent) == (kind, intent), result.text


def test_table_is_big_enough() -> None:
    assert len(TABLE) >= 60
    assert {kind for _, kind, _ in TABLE} >= set(RouteKind) - {RouteKind.EMPTY}


@pytest.mark.parametrize(
    ("text", "place"),
    [
        ("I'm going to Redmond Town Center.", "redmond town center"),
        ("Hey DrivePilot, take me to Seattle Children's Hospital!", "seattle children's hospital"),
        ("Navigate to 1600 Amphitheatre Parkway, Mountain View", "1600 amphitheatre parkway mountain view"),
        ("I'm going to McDonald's", "mcdonald's"),
    ],
)
def test_destination_place_is_extracted(text: str, place: str) -> None:
    assert route(text).place == place


def test_wake_word_and_filler_are_removed() -> None:
    assert normalize("Hey, DrivePilot! Um, who's first?") == "who is first"
    assert normalize("OK drive pilot so can I go") == "can i go"
    assert normalize("Hey Drive-Pilot: is it OK to merge?") == "is it ok to merge"
    assert strip_wake_word("Hey DrivePilot, what's the speed limit?") == "what's the speed limit?"


def test_model_gets_the_question_as_spoken() -> None:
    r = route("Hey DrivePilot, what does that sign say, like, exactly?")
    assert r.kind is RouteKind.MODEL and r.question == "what does that sign say, like, exactly?"


@pytest.mark.parametrize("text", ["", "Hey DrivePilot", "Hey DrivePilot...", "um", "  "])
def test_nothing_said(text: str) -> None:
    assert route(text).kind is RouteKind.EMPTY
