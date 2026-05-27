import re
import json
from typing import List, Dict
from pathlib import Path

# ------------------------------------------------------------------
# REGEX PATTERNS
# ------------------------------------------------------------------

# Matches standard screenplay scene headings like:
# INT. ROOM - NIGHT
# EXT. STREET - DAY
SCENE_PATTERN = re.compile(r'^(INT\.|EXT\.).+', re.IGNORECASE)

# Matches speaker names written in all caps
# e.g., JOHN, MARY JANE
SPEAKER_PATTERN = re.compile(r'^[A-Z][A-Z ]+$')


# ------------------------------------------------------------------
# MAIN PARSER FUNCTION
# ------------------------------------------------------------------

def parse_screenplay(text: str) -> List[Dict]:
    """
    Converts raw screenplay text into structured units.

    Each unit is one of:
    - Dialogue (spoken by a character)
    - Action (descriptive text)
    
    Returns:
        A list of dictionaries, each representing one unit.
    """

    units = []                 # Final list of parsed units
    current_scene = None       # Keeps track of which scene we are in

    # Split screenplay into non-empty lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    i = 0                      # Line pointer

    # Process screenplay line by line
    while i < len(lines):
        line = lines[i]

        # ----------------------------------------------------------
        # 1. SCENE HEADING
        # ----------------------------------------------------------
        # If the line matches INT./EXT., update the current scene
        if SCENE_PATTERN.match(line):
            current_scene = line
            i += 1
            continue

        # ----------------------------------------------------------
        # 2. DIALOGUE BLOCK
        # ----------------------------------------------------------
        # Speaker names are in ALL CAPS
        if SPEAKER_PATTERN.match(line):
            speaker = line
            dialogue = []
            i += 1

            # Collect all dialogue lines until:
            # - another speaker appears
            # - a new scene begins
            while (
                i < len(lines)
                and not SPEAKER_PATTERN.match(lines[i])
                and not SCENE_PATTERN.match(lines[i])
            ):
                dialogue.append(lines[i])
                i += 1

            # Save dialogue unit
            units.append({
                "scene": current_scene,
                "speaker": speaker,
                "text": " ".join(dialogue),
                "type": "dialogue"
            })
            continue

        # ----------------------------------------------------------
        # 3. ACTION LINE
        # ----------------------------------------------------------
        # Anything that is not a scene heading or dialogue
        # is treated as an action description
        units.append({
            "scene": current_scene,
            "speaker": None,
            "text": line,
            "type": "action"
        })
        i += 1

    return units


# ------------------------------------------------------------------
# SAVE PARSED OUTPUT
# ------------------------------------------------------------------

def save_parsed_script(units: List[Dict], output_path: str) -> None:
    """
    Saves parsed screenplay units into a JSON file.

    This makes the data reusable for:
    - Feature extraction
    - Model training
    - Reproducibility
    """

    output_path = Path(output_path)

    # Create folders if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON with indentation for readability
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(units, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# SCRIPT ENTRY POINT (FOR TESTING)
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Input screenplay (raw text)
    input_path = "data/raw_scripts/scenes/godfather_scene_01.txt"

    # Output structured JSON
    output_path = "data/processed_scripts/sample.json"

    # Read screenplay
    with open(input_path, "r", encoding="utf-8") as f:
        script_text = f.read()

    # Parse screenplay into structured units
    units = parse_screenplay(script_text)

    # Save parsed data
    save_parsed_script(units, output_path)

    print(f"Parsed script saved to {output_path}")
