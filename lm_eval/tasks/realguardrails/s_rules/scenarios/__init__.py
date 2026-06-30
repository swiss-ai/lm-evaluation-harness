# Vendored from https://github.com/normster/llm_rules at SHA
# f627e569146015d7fd6f200bb758b2591f9eb6c6 (2025-02-12).
# Licensed under the Apache License, Version 2.0. Imports adapted to the
# lm_eval package layout; otherwise verbatim.

from . import assets, utils
from .games import (
    AnimalSounds,
    BinarySearch,
    ForbiddenWord,
    Questions,
    RockPaperScissors,
    SimonSays,
)
from .scenario import BaseScenario, Result
from .security import (
    AccessControl,
    Authentication,
    Commitment,
    Confidentiality,
    DiningCryptographers,
    Encryption,
    Integrity,
    Millionaires,
)


SECURITY_SCENARIOS = [
    Encryption,
    Integrity,
    AccessControl,
    Commitment,
    Confidentiality,
    Authentication,
    Millionaires,
    DiningCryptographers,
]

GAME_SCENARIOS = [
    ForbiddenWord,
    RockPaperScissors,
    AnimalSounds,
    Questions,
    BinarySearch,
    SimonSays,
]

SCENARIO_CLASSES = [
    Encryption,
    Integrity,
    AccessControl,
    Commitment,
    Confidentiality,
    Authentication,
    Millionaires,
    DiningCryptographers,
    ForbiddenWord,
    RockPaperScissors,
    AnimalSounds,
    Questions,
    BinarySearch,
    SimonSays,
]

SCENARIOS = {s.__name__: s for s in SCENARIO_CLASSES}
