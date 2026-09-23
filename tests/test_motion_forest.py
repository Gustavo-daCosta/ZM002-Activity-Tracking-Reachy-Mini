"""The exported numpy forest must predict exactly what the scikit-learn model predicts."""

import numpy as np
import pytest

from core.motion.forest import NumpyForest, export_forest, load_forest
from core.motion.features import FEATURE_NAMES


def make_model(seed=0):
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(120, len(FEATURE_NAMES)))
    y = (x[:, 0] + rng.normal(scale=0.3, size=120) > 0).astype(int)
    return RandomForestClassifier(n_estimators=12, max_depth=4, random_state=0).fit(x, y), x


def test_numpy_forest_matches_sklearn(tmp_path):
    model, x = make_model()
    path = tmp_path / "forest.npz"
    export_forest(model, path)

    forest = load_forest(path)
    expected = model.predict_proba(x)[:, list(model.classes_).index(1)]
    got = np.array([forest.wave_probability(row) for row in x])

    assert got == pytest.approx(expected, abs=1e-6)


def test_export_keeps_the_feature_contract(tmp_path):
    model, _ = make_model()
    path = tmp_path / "forest.npz"
    export_forest(model, path)

    forest = load_forest(path)
    assert forest.feature_names == list(FEATURE_NAMES)


def test_load_rejects_a_forest_trained_on_other_features(tmp_path):
    model, _ = make_model()
    path = tmp_path / "forest.npz"
    export_forest(model, path, feature_names=["a", "b"])

    with pytest.raises(ValueError, match="retrain"):
        load_forest(path)


def test_single_row_prediction_is_deterministic(tmp_path):
    model, x = make_model()
    path = tmp_path / "forest.npz"
    export_forest(model, path)
    forest = load_forest(path)

    assert forest.wave_probability(x[0]) == forest.wave_probability(x[0])


def test_numpy_forest_rejects_a_wrong_sized_sample():
    forest = NumpyForest(
        children_left=[np.array([-1])], children_right=[np.array([-1])],
        feature=[np.array([-2])], split_threshold=[np.array([0.0])],
        class_proba=[np.array([[0.0, 1.0]])], feature_names=list(FEATURE_NAMES),
    )
    with pytest.raises(ValueError, match="9 features"):
        forest.wave_probability(np.zeros(3))


def test_threshold_survives_the_export(tmp_path):
    model, _ = make_model()
    path = tmp_path / "forest.npz"
    export_forest(model, path, threshold=0.62)

    assert load_forest(path).threshold == pytest.approx(0.62)


def test_default_threshold_is_one_half(tmp_path):
    model, _ = make_model()
    path = tmp_path / "forest.npz"
    export_forest(model, path)

    assert load_forest(path).threshold == pytest.approx(0.5)


def test_multiclass_round_trip(tmp_path):
    """A three-class forest survives the numpy export and predicts the same class as sklearn."""
    from sklearn.ensemble import RandomForestClassifier

    from core.motion.forest import export_multiclass_forest, load_forest

    rng = np.random.default_rng(0)
    names = ("none", "wave", "squat")
    # Three well-separated blobs, one per class, in a 4-feature space.
    centers = {"none": [0, 0, 0, 0], "wave": [5, 0, 0, 0], "squat": [0, 5, 0, 0]}
    samples, labels = [], []
    for name, center in centers.items():
        samples.append(rng.normal(center, 0.3, size=(60, 4)))
        labels += [name] * 60
    X = np.vstack(samples)
    model = RandomForestClassifier(n_estimators=8, random_state=0).fit(X, labels)

    features = ("f0", "f1", "f2", "f3")
    path = export_multiclass_forest(
        model, tmp_path / "actions.npz", feature_names=features, window_s=3.0,
        thresholds={name: 0.4 for name in names}, classes=names,
    )
    forest = load_forest(path, feature_names=features, window_s=3.0)

    assert forest.classes == list(names)
    assert forest.thresholds == {name: 0.4 for name in names}
    for sample, expected in ((centers["wave"], "wave"), (centers["squat"], "squat"),
                             (centers["none"], "none")):
        probabilities = forest.probabilities(sample)
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        assert max(probabilities, key=probabilities.get) == expected
        assert forest.predict(sample)[0] == expected


def test_predict_falls_back_to_none_below_the_class_floor(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    from core.motion.forest import export_multiclass_forest, load_forest

    # Overlapping enough (std 1.0, centers 2.2 apart) that an 8-tree forest lands short of
    # unanimous: seed 2 was checked to give the forest ~0.875 confidence on [2.2, 0], not 1.0
    # (which well-separated blobs would produce, making the 0.99 floor untestable).
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal([0, 0], 1.0, (40, 2)), rng.normal([2.2, 0], 1.0, (40, 2))])
    labels = ["none"] * 40 + ["wave"] * 40
    model = RandomForestClassifier(n_estimators=8, random_state=0).fit(X, labels)
    path = export_multiclass_forest(
        model, tmp_path / "f.npz", feature_names=("a", "b"), window_s=3.0,
        thresholds={"none": 0.0, "wave": 0.99}, classes=("none", "wave"),
    )
    forest = load_forest(path, feature_names=("a", "b"), window_s=3.0)
    # A confident wave, but the floor is higher than any forest of 8 trees will reach.
    assert 0.8 < forest.probabilities([2.2, 0])["wave"] < 0.99
    assert forest.predict([2.2, 0]) == ("none", 0.0)


def test_binary_wave_export_still_loads(tmp_path):
    """The wave model on the robot must keep working without being retrained."""
    from sklearn.ensemble import RandomForestClassifier

    from core.motion.features import FEATURE_NAMES
    from core.motion.forest import export_forest, load_forest
    from core.motion.window import WINDOW_S

    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=6, random_state=0).fit(X, y)
    path = export_forest(model, tmp_path / "wave.npz", threshold=0.6)

    forest = load_forest(path)
    assert forest.threshold == pytest.approx(0.6)
    assert forest.classes == [0, 1]
    sample = X[np.argmax(X[:, 0])]
    assert forest.wave_probability(sample) == pytest.approx(
        model.predict_proba(sample[None])[0][list(model.classes_).index(1)], abs=1e-9
    )
