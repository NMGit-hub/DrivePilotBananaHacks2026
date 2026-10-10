"""Router table: phrasings as Whisper transcribes them → handler (BUILD_PLAN M3: >= 60 phrasings)."""

from __future__ import annotations

import pytest

from voice.router import RouteKind, normalize, route, strip_wake_word

P = RouteKind.PARKING
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
    ("Can you repeat that?", C, "repeat"),  # the model has no memory of what was said
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
    # --- find me a parking spot (V2)
    ("Hey DrivePilot, find me a parking spot.", P, "find"),
    ("Help me find a spot.", P, "find"),
    ("Can you help me find parking?", P, "find"),
    ("Where can I park?", P, "find"),
    ("Look for an open space.", P, "find"),
    ("Keep an eye out for parking.", P, "find"),
    ("Are there any open spots?", P, "find"),
    ("Find me street parking.", P, "find_street"),
    ("Look for a spot on the street.", P, "find_street"),
    ("Stop looking.", P, "stop"),
    ("Stop looking for parking.", P, "stop"),
    ("I found a spot.", P, "stop"),
    ("We're parked.", P, "stop"),
    ("Can I park here?", M, "scene"),  # a sign question, not the parking watch
    ("Is this a parking spot?", M, "scene"),
    ("What does that parking sign say?", M, "scene"),
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
    ("Should I wait?", R, "should_wait"),  # not "can I go?": yes/no would mean the opposite
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
    ("How far to my turn?", N, "turn_distance"),
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
    ("Should I move over to the left?", G, "gap_check"),  # a lane change now (no route mentioned)
    # --- everything else: the vision model
    ("What's the speed limit?", M, "scene"),
    ("What did that sign say?", M, "memory"),  # V2-M4: a sign we passed: remembered frames + close-ups
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


# ------------------------------------------------------------------------------- adversarial table (M3 review)

# Realistic variants a reviewer wrote without looking at the patterns. A miss in the first block means a
# split-second question answered while moving, so it is the most important part of this table.
ADVERSARIAL: list[tuple[str, RouteKind, str]] = [
    *[
        (text, G, "gap_check")
        for text in (
            "Safe to merge?",
            "OK to merge?",
            "Clear to merge?",
            "Good to merge?",
            "Is it safe for me to merge?",
            "Is my lane clear?",
            "Is the lane next to me clear?",
            "Is the lane to my left clear?",
            "Is the right side clear?",
            "Can I get over?",
            "Can I move over?",
            "Can I move into the left lane?",
            "Can I change to the left lane?",
            "Can I get in the left lane?",
            "Can I go left now?",
            "Can I turn left now?",
            "Can I still make it?",
            "Will I make the light?",
            "Am I going to make the light?",
            "Am I going to make it?",
            "Do I have time to make the light?",
            "Should I stop or go?",
            "Should I stop?",
            "Should I brake?",
            "Is there room to merge?",
            "Do I have room to merge?",
            "Am I clear to merge?",
            "Am I clear?",
            "Is anyone on my left?",
            "Is there a car on my left?",
            "Anyone next to me?",
            "Is the road ahead clear?",
            "Is it safe?",
            "Is it safe now?",
            "Can I squeeze in?",
            "Is the left lane open?",
            "Is that car going to stop?",
            "Is he going to let me in?",
            "Can I make this left?",
            "Can I go before that car?",
            "Okay, so can I merge now?",
            "Is it okay if I merge?",
            "Can I run this yellow?",
            "Can I go through the yellow?",
            "Can I pull out?",
            "Is the intersection clear?",
            "Can I make the turn before that car?",
            "Merge now?",
            "Is it safe to get on the highway?",
            "OK to go?",
            "Good to go?",
            "Am I OK to go?",
            "Is it OK for me to go?",
            "I'm going to merge, is that ok?",
            "I'm going to try to make the light",
            "I'm going to cut in front of this truck",
            "Right lane clear?",
            "Right side clear?",
        )
    ],
    *[
        (text, G, "gap_watch")
        for text in (
            "Tell me when it's safe to merge",
            "Let me know when I can merge",
            "Tell me when it's clear",
            "Tell me when the lane is clear",
            "Tell me when I can turn left",
        )
    ],
    *[
        (text, R, "can_i_go")
        for text in (
            "Okay, so can I go?",
            "Um so can I go now?",
            "Can I go now, DrivePilot?",
            "Can I go here?",
            "Can I go yet or not?",
            "Can I still go?",
            "When's my turn?",
            "Is it me next?",
            "its my turn?",
            "Hey Dr Pilot, can I go?",
            "Hey Drive Pylot can I go",
            "Can I go, or what?",
            "Can I go now that the truck left?",
        )
    ],
    *[
        (text, R, "who_first")
        for text in (
            "Which car goes first?",
            "Which car got here first?",
            "Was I first?",
            "Did I get here first?",
            "Does the truck go first?",
            "Do I go before the truck?",
        )
    ],
    ("How many cars go before me?", R, "position"),
    ("Should I wait for the pedestrian?", R, "should_wait"),
    ("Should I keep waiting?", R, "should_wait"),
    ("Can you watch the light?", W, "light_watch"),
    ("Can you tell me when it's green?", W, "light_watch"),
    ("Tell me when the car ahead goes", W, "go_watch"),
    *[
        (text, N, "next_step")
        for text in (
            "Tell me when to turn",
            "Tell me where to go",
            "Show me where to go",
            "Tell me which way to go",
            "Tell me about the next turn",
            "Let me know my next turn",
            "Tell me when my turn is coming up",
            "Let me know if I need to go left",
            "Is it the next light?",
        )
    ],
    ("Remind me to go to the store", M, "scene"),
    ("Can you tell me if I can turn right here?", M, "scene"),
    ("Tell me what turns are allowed here", M, "scene"),
    *[
        (text, C, "stop")
        for text in ("Stop, stop", "Stop now", "Stop that", "Shush", "Quiet please", "Okay, stop.", "No, stop")
    ],
    *[
        (text, C, "normal")
        for text in ("Turn off quiet mode", "Exit quiet mode", "Disable quiet mode", "Back to normal")
    ],
    *[(text, C, "quiet") for text in ("Turn off alerts", "Turn off the alerts", "Enable quiet mode")],
    *[
        (text, C, "repeat")
        for text in (
            "Can you repeat that?",
            "Could you say that again?",
            "What?",
            "What was that?",
            "Sorry?",
            "Huh?",
            "Repeat",
        )
    ],
    *[(text, C, "cancel") for text in ("Stop watching the light", "Cancel the alert", "Don't tell me")],
    ("Stop sign ahead?", M, "scene"),
    ("Stop sign?", M, "scene"),
    ("How much longer until we get there?", N, "eta"),
    ("How much longer do we have?", N, "eta"),
    ("Am I next to a fire hydrant?", M, "scene"),
    ("Am I up against the curb?", M, "scene"),
    ("Who's next to me?", M, "scene"),
    ("How long has the car on the left been waiting?", M, "scene"),
    ("How far is the car on the left?", M, "scene"),
    ("Should I change lanes for my exit?", L, "lane"),
    ("Should I switch lanes to get to 520?", L, "lane"),
]


@pytest.mark.parametrize(("text", "kind", "intent"), ADVERSARIAL, ids=[t for t, _, _ in ADVERSARIAL])
def test_router_adversarial(text: str, kind: RouteKind, intent: str) -> None:
    result = route(text)
    assert (result.kind, result.intent) == (kind, intent), result.text


@pytest.mark.parametrize(
    ("text", "place"),
    [
        ("Take me home", "home"),
        ("Navigate home", "home"),
        ("Go home", "home"),
        ("I'm going to Maple Lane", "maple lane"),
        ("Going to Park Avenue", "park avenue"),
        ("Going to Stop and Shop", "stop and shop"),
        ("Heading to Seattle, how long?", "seattle"),
        ("Take us to the next rest stop", "the next rest stop"),
        ("I'm driving to work", "work"),
    ],
)
def test_destination_phrasings(text: str, place: str) -> None:
    result = route(text)
    assert result.kind is RouteKind.DESTINATION and result.place == place, result


@pytest.mark.parametrize(
    "text",
    [
        "I'm going to speed up",
        "I'm going to slow down",
        "I'm going to overtake",
        "Drive to the stop line",
        "I'm heading to the exit",
        "I'm going to turn left here, is that allowed?",
    ],
)
def test_not_destinations(text: str) -> None:
    assert route(text).kind is not RouteKind.DESTINATION


# Real Whisper transcripts from the synthetic voice benchmark (runs/*_voice) that used to reach the vision model:
# junk before the question (a garbled wake word, a stray word, a lead-in) and common mishearings.
TRANSCRIPTS: list[tuple[str, RouteKind, str]] = [
    ("Hi-LUT, stop talking.", C, "stop"),
    ("Get it. Stop now.", C, "stop"),
    ("hey drive, pilot, quiet mode.", C, "quiet"),
    ("turn off alert.", C, "quiet"),
    ("Hi Alact. Can you watch the light?", W, "light_watch"),
    ("Hi, lad, when can I go?", R, "can_i_go"),
    ("can't I go now that the truck left?", R, "can_i_go"),
    ("should I break?", G, "gap_check"),
    ("The light just changed, should I go?", R, "can_i_go"),
    ("stop navigating", N, "cancel_route"),
    ("How long until my next turn?", N, "turn_distance"),
    ("How far to my destination?", N, "eta"),
]


@pytest.mark.parametrize(("text", "kind", "intent"), TRANSCRIPTS, ids=[t for t, _, _ in TRANSCRIPTS])
def test_router_real_transcripts(text: str, kind: RouteKind, intent: str) -> None:
    result = route(text)
    assert (result.kind, result.intent) == (kind, intent), result.text


@pytest.mark.parametrize(
    "text",
    [
        "What does that sign say, the one on the right?",
        "Is that a bike lane, or a parking lane?",
        "Why can't I go?",
        "The car on the left, is it parked?",
    ],
)
def test_dropping_a_lead_in_keeps_real_scene_questions(text: str) -> None:
    assert route(text).kind is RouteKind.MODEL


def test_a_lead_in_never_hides_a_gap_check() -> None:
    # The whole text is checked first: a split-second question stays a gap check whatever comes before it.
    assert route("Okay, the truck is slow, can I pass it now?").kind is RouteKind.GAP_CHECK


@pytest.mark.parametrize(
    ("text", "kind", "intent"),
    [
        ("Can I pass the school bus?", G, "school_bus"),
        ("should I go around that school bus", G, "school_bus"),
        ("is it ok to overtake the school bus", G, "school_bus"),
        ("why is that school bus stopped?", M, "scene"),  # not about passing: the model describes it
        ("can I pass this truck?", G, "gap_check"),  # any other vehicle: a gap judgment
    ],
)
def test_passing_a_school_bus_is_a_fixed_rule(text: str, kind: RouteKind, intent: str) -> None:
    result = route(text)
    assert (result.kind, result.intent) == (kind, intent)
