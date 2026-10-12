"""Everything the coach says, in one place, so the tone can be tuned without touching the logic.

The listener is most likely a child: short sentences, warm and playful, never blaming.
"""

SPOKEN = {"squat": "squats", "arm_raise": "arm raises", "side_bend": "side bends"}
HOW = {
    "squat": "Bend your knees like you're sitting on a tiny chair, then stand up tall!",
    "arm_raise": "Lift both arms out to the sides like a bird, then bring them down!",
    "side_bend": "Hands on your hips, then lean to one side and the other, like a tree in the wind!",
}
# "none" rounds: ordinary things that are not an exercise (the collect kit's NONE_HINTS, for kids).
FREE_TIME = ("Stand still like a statue!", "Scratch your head!", "Cross your arms!", "Look around the room!",
             "Take a little step!", "Wiggle your fingers!")
PARTS = {"head": "your head", "hips": "your tummy", "knees": "your knees", "ankles": "your feet"}
PRAISE = ("Yay!", "Awesome!", "You're a star!", "Super!", "Wow, great job!", "High five!")
THINKING = ("Hmm, let me think...", "I'm learning your squats...", "Now your arm raises...",
            "My robot brain is working hard!", "Almost there, thanks for waiting!")


def intro(name):
    return (f"Hi {name}! I'm Reachy! Today you are going to be my teacher. "
            "You'll show me three moves: squats, arm raises and side bends. "
            "Then I'll learn them, and we'll play together!")


def step_back(parts):
    names = [PARTS[part] for part in parts]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"Can you step back a little? I can't see {listed}!"


def ready():
    return "Perfect! Stay right there!"


def welcome_back():
    return "There you are! Let's keep going!"


def gone():
    return "Hey, where did you go? Come back so I can see you!"


def round_intro(name, free_index):
    if name == "none":
        return f"Now a little break! {FREE_TIME[free_index % len(FREE_TIME)]}"
    return f"Next: {SPOKEN[name]}! {HOW[name]}"


def go():
    return "Go!"


def praise(rng):
    return rng.choice(PRAISE)


def halfway():
    return "We're halfway there!"


def redo():
    return "Oops, I missed that one! Let's do it again!"


def move_on():
    return "That's okay! Let's try the next one!"


def thanks_training():
    return "Thank you, teacher! Now let me think really hard to learn your moves. It takes a little while!"


def thinking(index):
    return THINKING[index % len(THINKING)]


def training_failed():
    return "Hmm, my brain got a bit tangled. Let's play with what I already know!"


def learned():
    return "I learned your moves! Now let's try them together!"


def lets_play(name):
    return f"Hi {name}! I remember your moves. Let's play!"


def save_failed():
    return "Oh no, I couldn't save our game. Please ask a grown-up for help!"
