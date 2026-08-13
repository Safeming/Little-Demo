from pathlib import Path


def test_body_models_root_can_be_overridden(monkeypatch, tmp_path):
    from utils.general_utils import get_body_model_misc_path

    monkeypatch.setenv("BODY_MODELS_ROOT", str(tmp_path / "models"))

    assert Path(get_body_model_misc_path("faces.npz")) == tmp_path / "models" / "misc" / "faces.npz"
