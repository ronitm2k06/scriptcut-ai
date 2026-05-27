from src.features.narrative_features import extract_scene_features
from src.features.scene_classifier import classify_scene


def build_scene_dataset(parsed_scenes: dict) -> list:
    """
    Build a dataset from parsed script scenes.

    parsed_scenes format:
    {
        "scene_id": [scene_units...]
    }
    """

    dataset = []

    for scene_id, scene_units in parsed_scenes.items():
        features = extract_scene_features(scene_units)
        if not features:
            continue

        scene_type = classify_scene(features)

        # Target label (editorial pacing)
        if scene_type == "static_dialogue":
            pacing_class = "slow"
        else:
            pacing_class = "fast"

        dataset.append({
            "scene_id": scene_id,
            "features": features,
            "scene_type": scene_type,
            "target": {
                "pacing_class": pacing_class
            }
        })

    return dataset


def group_units_by_scene(parsed_units: list) -> dict:
    """
    Group flat parsed screenplay units by their 'scene' field.

    Returns a dict: {scene_heading: [units...]}
    """
    grouped = {}
    current_scene = "PROLOGUE"  # Fallback if first units have no scene heading

    for unit in parsed_units:
        scene = unit.get("scene")
        if scene:
            current_scene = scene

        if current_scene not in grouped:
            grouped[current_scene] = []

        grouped[current_scene].append(unit)

    return grouped
