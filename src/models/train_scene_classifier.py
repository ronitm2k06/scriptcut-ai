import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_scene_classifier(dataset: list):
    """
    Train a logistic regression model to classify scene type.
    """

    X = []
    y = []

    for row in dataset:
        features = row["features"]

        X.append([
            features["dialogue_ratio"],
            features["action_density"],
            features["avg_dialogue_length"],
            features["speaker_switch_rate"],
            features["num_unique_speakers"],
            features["scene_length_lines"],
            features["sentiment_variance"],
        ])

        y.append(
            1 if row["scene_type"] == "dynamic" else 0
        )

    X = np.array(X)
    y = np.array(y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    pipeline.fit(X, y)

    return pipeline
