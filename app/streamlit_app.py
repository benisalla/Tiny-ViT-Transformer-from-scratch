"""Streamlit demo: upload an image, see the class, the confidence and where the ViT looked.

pip install -e '.[app]'
streamlit run app/streamlit_app.py -- --checkpoint checkpoints/run/best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

from tiny_vit.data import SHAPE_CLASSES, ShapeRenderer
from tiny_vit.predict import Predictor, overlay_heatmap

TITLE = "Tiny-ViT Image Classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/run/best.pt"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args(sys.argv[1:])


@st.cache_resource(show_spinner="Loading the model...")
def load_predictor(checkpoint: str, device: str) -> Predictor:
    return Predictor.from_checkpoint(checkpoint, device)


def classify(image: Image.Image, predictor: Predictor, top_k: int) -> list[tuple[str, float]]:
    """[(class, probability)] best first."""
    return predictor.predict(image, top_k=top_k)


def attention_overlay(image: Image.Image, predictor: Predictor, alpha: float) -> Image.Image:
    """The image with the attention rollout drawn in red on top."""
    return overlay_heatmap(image, predictor.attention_heatmap(image), alpha)


def pretty(name: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato · Late blight' (PlantVillage folder names)."""
    return " · ".join(part.replace("_", " ").strip() for part in name.split("___"))


def sidebar(args: argparse.Namespace) -> dict:
    st.sidebar.header("Model")
    settings = {
        "checkpoint": st.sidebar.text_input("Checkpoint", str(args.checkpoint)),
        "device": args.device,
    }
    st.sidebar.header("Display")
    settings["top_k"] = st.sidebar.slider("Classes to show", 1, 10, 5)
    settings["show_attention"] = st.sidebar.checkbox("Show attention map", value=True)
    settings["alpha"] = st.sidebar.slider("Attention overlay strength", 0.0, 1.0, 0.55, 0.05)
    return settings


def model_card(predictor: Predictor) -> None:
    config, data = predictor.model.config, predictor.data_config
    with st.expander("About this model"):
        st.markdown(
            f"- **Dataset:** `{data.dataset}` · {len(predictor.class_names)} classes\n"
            f"- **Input:** {config.image_size}×{config.image_size} px, cut into "
            f"{config.num_patches} patches of {config.patch_size}×{config.patch_size}\n"
            f"- **Transformer:** {config.n_layers} layers · {config.n_heads} heads · "
            f"width {config.d_model} · {predictor.model.num_parameters() / 1e6:.2f}M parameters\n"
            f"- **Classes:** " + ", ".join(pretty(c) for c in predictor.class_names)
        )


def pick_image(predictor: Predictor) -> Image.Image | None:
    """An uploaded image, or (for a shapes model) a freshly generated one."""
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if upload is not None:
        return Image.open(upload).convert("RGB")
    if predictor.class_names == list(SHAPE_CLASSES):
        if st.button("Generate a random shape"):
            st.session_state["shape_index"] = random.randrange(10**9)
        if "shape_index" in st.session_state:
            return ShapeRenderer(seed=12345)(st.session_state["shape_index"])
    return None


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title=TITLE, page_icon="🔍", layout="centered")
    st.title(TITLE)
    st.caption("A Vision Transformer written from scratch in PyTorch.")
    settings = sidebar(args)

    if not Path(settings["checkpoint"]).exists():
        st.error(
            f"Checkpoint not found: {settings['checkpoint']}. Train one with `vit-train` "
            "(see the README), then pass it with `-- --checkpoint path/to/best.pt`."
        )
        return
    predictor = load_predictor(settings["checkpoint"], settings["device"])
    model_card(predictor)

    image = pick_image(predictor)
    if image is None:
        st.info("Upload an image to classify it.")
        return

    ranked = classify(image, predictor, settings["top_k"])
    best, confidence = ranked[0]
    st.subheader(f"{pretty(best)}  ({confidence:.0%})")

    left, right = st.columns(2)
    left.image(image, caption="Input", width="stretch")
    if settings["show_attention"]:
        right.image(
            attention_overlay(image, predictor, settings["alpha"]),
            caption="Where the model looked (attention rollout)",
            width="stretch",
        )
    st.markdown("**Top predictions**")
    for name, prob in ranked:
        st.progress(prob, text=f"{pretty(name)}: {prob:.1%}")


if __name__ == "__main__":
    main()
