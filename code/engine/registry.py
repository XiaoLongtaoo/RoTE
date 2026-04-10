"""Registry for backbone-specific runners."""

from __future__ import annotations

DISPLAY_NAMES = {
    ("sasrec", "baseline"): "SASRec",
    ("sasrec", "rote"): "RoTE-SASRec",
    ("rpg", "baseline"): "RPG",
    ("rpg", "rote"): "RoTE-RPG",
}


def get_runner(config):
    backbone = config["backbone"]
    if backbone == "sasrec":
        from engine.sasrec_runner import SASRecRunner

        return SASRecRunner
    if backbone == "rpg":
        from engine.rpg_runner import RPGRunner

        return RPGRunner
    raise ValueError(f"Unsupported backbone: {backbone}")


def get_display_name(config):
    return DISPLAY_NAMES.get((config["backbone"], config["variant"]), f"{config['backbone']}-{config['variant']}")
