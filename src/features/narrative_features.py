from collections import Counter
import numpy as np


def extract_scene_features(scene_units: list) -> dict:
    """
    Extract narrative-level features from a parsed script scene.

    scene_units: list of dicts like:
    {
        'type': 'dialogue' | 'action',
        'speaker': 'NAME' | None,
        'text': '...'
    }
    """

    total_lines = len(scene_units)
    if total_lines == 0:
        return {}

    # Separate dialogue and action
    dialogue_units = [u for u in scene_units if u["type"] == "dialogue"]
    action_units = [u for u in scene_units if u["type"] == "action"]

    num_dialogue = len(dialogue_units)
    num_action = len(action_units)

    # --- Core ratios ---
    dialogue_ratio = num_dialogue / total_lines
    action_density = num_action / total_lines

    # --- Dialogue length ---
    dialogue_lengths = [
        len(u["text"].split()) for u in dialogue_units
    ]

    avg_dialogue_length = (
        np.mean(dialogue_lengths) if dialogue_lengths else 0
    )

    # --- Speaker dynamics ---
    speakers = [u["speaker"] for u in dialogue_units if u["speaker"]]
    num_unique_speakers = len(set(speakers))

    speaker_switches = sum(
        1 for i in range(1, len(speakers))
        if speakers[i] != speakers[i - 1]
    )

    speaker_switch_rate = (
        speaker_switches / max(len(speakers) - 1, 1)
    )

    # --- Scene scale ---
    scene_length_lines = total_lines

    # --- Sentiment proxy (simple but effective) ---
    # Using punctuation & emphasis as emotional instability proxy
    sentiment_variance = sum(
        u["text"].count("!") + u["text"].count("?")
        for u in dialogue_units
    ) / max(num_dialogue, 1)

    return {
        "dialogue_ratio": dialogue_ratio,
        "action_density": action_density,
        "avg_dialogue_length": avg_dialogue_length,
        "speaker_switch_rate": speaker_switch_rate,
        "num_unique_speakers": num_unique_speakers,
        "scene_length_lines": scene_length_lines,
        "sentiment_variance": sentiment_variance,
    }
