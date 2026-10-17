"""The Streamlit web demo and the Kaggle / Colab notebook."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import write_checkpoint
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
NOTEBOOK = ROOT / "notebooks" / "train_colab_kaggle.ipynb"


# --- web app ----------------------------------------------------------------------


def load_app():
    pytest.importorskip("streamlit")
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_classifies_and_draws_attention(tmp_path: Path) -> None:
    app = load_app()
    predictor = app.Predictor.from_checkpoint(write_checkpoint(tmp_path), "cpu")
    image = Image.new("RGB", (90, 60), "white")
    ranked = app.classify(image, predictor, top_k=2)
    assert len(ranked) == 2 and isinstance(ranked[0][0], str)
    assert app.attention_overlay(image, predictor, 0.5).size == image.size


def test_pretty_class_names() -> None:
    app = load_app()
    assert app.pretty("Tomato___Late_blight") == "Tomato · Late blight"
    assert app.pretty("circle") == "circle"


def run_page(monkeypatch: pytest.MonkeyPatch, checkpoint: str):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["app", "--checkpoint", checkpoint, "--device", "cpu"])
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_page_reports_a_missing_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    page = run_page(monkeypatch, "does/not/exist.pt")
    assert not page.exception
    assert page.title[0].value == "ViT Image Classifier"
    assert "Checkpoint not found" in page.error[0].value


def test_page_loads_a_checkpoint_and_waits_for_an_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    page = run_page(monkeypatch, str(write_checkpoint(tmp_path)))
    assert not page.exception
    assert not page.error
    assert "Upload an image" in page.info[0].value
    assert page.expander[0].label == "About this model"


def test_page_classifies_a_generated_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from vit_classifier.data import SHAPE_CLASSES

    page = run_page(monkeypatch, str(write_checkpoint(tmp_path, class_names=list(SHAPE_CLASSES))))
    page = page.button[0].click().run()
    assert not page.exception
    best = page.subheader[0].value  # e.g. "ring  (21%)"
    assert best.split()[0] in SHAPE_CLASSES and best.endswith("%)")
    assert len(page.get("progress")) == 5  # top-5 bars
    assert len(page.get("image")) == 2  # input + attention overlay


# --- notebook ---------------------------------------------------------------------


def build_notebook(out: Path):
    nbformat = pytest.importorskip("nbformat")
    subprocess.run(
        [sys.executable, str(ROOT / "notebooks" / "build_notebook.py"), str(ROOT), str(out)],
        check=True,
        capture_output=True,
    )
    return nbformat.read(out, as_version=4)


def test_notebook_embeds_the_current_package_code(tmp_path: Path) -> None:
    notebook = build_notebook(tmp_path / "nb.ipynb")
    written = {
        cell.source.split("\n", 1)[0].split()[-1]: cell.source.split("\n", 1)[1]
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.startswith("%%writefile")
    }
    package = ROOT / "src" / "vit_classifier"
    expected = {f"src/vit_classifier/{p.name}": p.read_text() for p in package.glob("*.py")}
    assert set(written) == set(expected), "a module is missing from the notebook"
    for name, source in expected.items():
        assert written[name] == source.strip(), f"{name} in the notebook differs from src/"


def test_committed_notebook_is_up_to_date(tmp_path: Path) -> None:
    # Fails when src/ changed but `python notebooks/build_notebook.py . notebooks/...`
    # was not re-run.
    nbformat = pytest.importorskip("nbformat")
    fresh = build_notebook(tmp_path / "nb.ipynb")
    committed = nbformat.read(NOTEBOOK, as_version=4)
    assert [c.source for c in committed.cells] == [c.source for c in fresh.cells]


def test_notebook_code_cells_are_valid_python(tmp_path: Path) -> None:
    for cell in build_notebook(tmp_path / "nb.ipynb").cells:
        if cell.cell_type == "code" and not cell.source.startswith("%%"):
            ast.parse(cell.source)
