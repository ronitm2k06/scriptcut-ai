def classify_scene(features: dict) -> str:
    """
    Rule-based scene type classifier.
    """

    if (
        features["dialogue_ratio"] > 0.65 and
        features["avg_dialogue_length"] > 12 and
        features["action_density"] < 0.25 and
        features["num_unique_speakers"] <= 4
    ):
        return "static_dialogue"

    return "dynamic"
