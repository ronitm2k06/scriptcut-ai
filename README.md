# 📝 ScriptCut AI: Screenplay Parser & Narrative Pacing Classifier

**ScriptCut AI** is an advanced Natural Language Processing (NLP) and Machine Learning system that parses raw screenplay scripts and classifies the dramatic pacing of each scene (slow vs. fast). By extracting narrative and statistical speech dynamics, it helps directors, editors, and writers analyze story flow, pacing transitions, and dialogue distribution.

---

## 🚀 Key Features

*   **🎬 Screenplay Parser:** Automatically parses raw screenplays (`.txt` files) into structured elements, classifying text lines into **Dialogue**, **Action descriptions (stage directions)**, and **Scene Headings**.
*   **📊 Narrative Feature Extraction:** Extracts high-level quantitative features from each scene, including:
    *   `dialogue_ratio`: Proportion of dialogue lines relative to the total scene.
    *   `action_density`: Stage direction frequency.
    *   `avg_dialogue_length`: Average words spoken per character line.
    *   `speaker_switch_rate`: Frequency of dialogue transitions between speakers.
    *   `num_unique_speakers`: Total distinct characters interacting.
    *   `scene_length_lines`: Physical size of the scene in lines.
    *   `sentiment_variance`: proxy measurement for emotional volatility.
*   **🧠 Dual Classification Models:**
    *   **Rule-Based Classifier:** Evaluates scenes using standard narrative pacing thresholds.
    *   **Logistic Regression Machine Learning Classifier:** Learns pacing classifications using feature scaling and standard predictive training pipelines.

---

## 📁 Directory Structure

```text
scriptcut-ai/
│
├── .gitignore                      # Keeps heavy media and interpreter caches off Git
├── README.md                       # Premium master project documentation
├── requirements.txt                # Light machine learning libraries
│
├── data/
│   ├── raw_scripts/
│   │   ├── The-Godfather-Script.txt # Full movie screenplay script
│   │   └── scenes/
│   │       └── godfather_scene_01.txt # Single-scene benchmark script
│   └── processed_scripts/          # Auto-generated JSON outputs from parser
│
└── src/
    ├── parser/
    │   └── screenplay_parser.py    # Screenplay structured parser
    ├── features/
    │   ├── narrative_features.py   # Narrative statistics extraction
    │   └── scene_classifier.py     # Rule-based pacing classifier
    ├── models/
    │   └── train_scene_classifier.py # Logistic Regression training module
    ├── utils/
    │   ├── dataset_builder.py      # Multi-scene dataset generator & scene grouper
    │   └── io.py                   # Clean JSON and text helper functions
    └── test_dataset.py             # Complete machine learning pipeline runner
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/RonitM2k06/scriptcut-ai.git
    cd scriptcut-ai
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate   # On Windows
    source .venv/bin/activate # On macOS/Linux
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 How to Run

### 1. Parse a Screenplay File
To run the screenplay parser test on a single scene, execute:
```bash
python -m src.parser.screenplay_parser
```
*Output JSON is saved to:* `data/processed_scripts/sample.json`

### 2. Run the Full ML Training & Pacing Pipeline
To parse the full Godfather movie script, segment it into all 13 scenes, extract features, and train the Machine Learning classifier to predict pacing:
```bash
python -m src.test_dataset
```

This pipeline will:
1. Parse `The-Godfather-Script.txt`.
2. Segment the text into **13 distinct visual scenes**.
3. Compute the 7-dimensional feature vectors.
4. Fit a standard **StandardScaler** and **LogisticRegression** model on the dataset.
5. Print out side-by-side comparative predictions (Rule-Based vs. ML predictions).

---

## 📊 Narrative Feature Formulation

*   **Dialogue Ratio:** 
    $$\text{Dialogue Ratio} = \frac{\text{Dialogue Lines}}{\text{Total Scene Lines}}$$
*   **Action Density:**
    $$\text{Action Density} = \frac{\text{Action Lines}}{\text{Total Scene Lines}}$$
*   **Speaker Switch Rate:**
    $$\text{Switch Rate} = \frac{\text{Speaker Switches}}{\text{Total Speaking Turns}}$$
