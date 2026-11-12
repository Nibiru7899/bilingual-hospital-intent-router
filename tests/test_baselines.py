from __future__ import annotations

import pytest

from ml.baselines import evaluate, load_model, predict, summarize, train
from ml.common import read_json


def test_metrics_against_known_confusion():
    metrics = summarize(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert metrics["accuracy"] == .75
    assert metrics["macro_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert metrics["macro_recall"] == .75
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + .8) / 2)


@pytest.mark.parametrize("kind", ["logistic", "svm"])
def test_training_round_trip_and_train_only_features(prepared, tmp_path, kind):
    run = tmp_path / kind
    manifest = train(prepared, run, kind)
    assert len(manifest["validation_trials"]) == 3
    assert not manifest["config"]["test_access_during_fit"]
    model, _ = load_model(run)
    features = model.named_steps["features"]
    vectorizers = [vectorizer for _, vectorizer in features.transformer_list] if kind == "logistic" else [features]
    for vectorizer in vectorizers:
        assert "testonlyword" not in vectorizer.vocabulary_
        assert "validationonlyword" not in vectorizer.vocabulary_
    if kind == "logistic":
        word = dict(features.transformer_list)["word"]
        assert "trainonlyword" in word.vocabulary_
        assert "validationonlyword" not in word.vocabulary_
        assert "testonlyword" not in word.vocabulary_
    first = predict(run, "invoice payment")
    assert first == predict(run, "invoice payment")
    assert not first["hospital_routing_validated"]
    assert not first["action_performed"]
    if kind == "svm":
        assert first["score_type"] == "decision_margin_not_probability"
        assert len(first["scores"]) == 2
    else:
        assert sum(first["scores"].values()) == pytest.approx(1.)
    result = evaluate(prepared, run)
    assert result["n"] == 12
    assert result["accuracy"] == 1.
    assert (run / "evaluation" / "confusion_matrix.png").exists()
    assert read_json(run / "evaluation" / "evaluation_manifest.json")["outputs_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        evaluate(prepared, run)
    with pytest.raises(ValueError, match="1-10000"):
        predict(run, " ")
    with (run / "model.joblib").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_model(run)
