# Inference

All three ways read the image size, normalisation and class names from the
checkpoint, so they always prepare images exactly like validation did.

## Command line

```bash
vit-predict --checkpoint checkpoints/plantvillage/best.pt --image leaf1.jpg leaf2.jpg --top-k 3
```

```
leaf1.jpg
   97.12%  Tomato___Late_blight
    1.40%  Tomato___Early_blight
    0.51%  Potato___Late_blight
```

`--json` prints the same as JSON; `--device cpu|cuda|mps` picks the device.

## Python

```python
from PIL import Image
from vit_classifier.predict import Predictor, overlay_heatmap

predictor = Predictor.from_checkpoint("checkpoints/plantvillage/best.pt")
image = Image.open("leaf.jpg")

predictor.predict(image, top_k=3)            # [("Tomato___Late_blight", 0.97), ...]
predictor.probabilities([image, image])      # (2, num_classes) tensor
heat = predictor.attention_heatmap(image)    # (8, 8) array in [0, 1]
overlay_heatmap(image, heat).save("where_it_looked.png")
```

`load_model(path)` returns the raw `(model, data_config, class_names)` if you
want to run the network yourself (`model(images)` returns logits).

## Web app (Streamlit)

```bash
pip install -e '.[app]'
streamlit run app/streamlit_app.py -- --checkpoint checkpoints/plantvillage/best.pt
```

(The `--` separates Streamlit's own flags from the app's.) The page shows:

- the predicted class and its probability, and the top-k classes as bars;
- the **attention rollout** heatmap: which patches the [CLS] token drew on, in
  red over the image (see [architecture.md](architecture.md#7-attention-rollout-what-the-app-shows));
- "About this model": dataset, input size, patch grid, depth, width,
  parameter count and the class list, all read from the checkpoint.

With a `shapes` checkpoint, a "Generate a random shape" button draws a new
synthetic image, so the demo works without any file to upload. The sidebar
lets you change the checkpoint path, the number of classes shown and the
overlay strength. A missing checkpoint gives a clear message instead of a
stack trace.

## Evaluate on the test split

```bash
vit-evaluate --checkpoint checkpoints/plantvillage/best.pt --split test \
             --report test_report.json --examples docs/examples/plantvillage
```

- Rebuilds the exact test split from the checkpoint's `DataConfig` (same
  `split_seed`): images the model never trained on and that were not used to
  pick `best.pt`. `--data-dir` points to the data if it moved.
- Prints accuracy, per-class **recall** (of the images of class c, how many
  were found) and **precision** (of the images predicted c, how many were
  right), and the confusion matrix (row = true class).
- `--report` saves the numbers as JSON; `--examples DIR` writes thumbnails and
  a Markdown table of `--num-examples` test images spread evenly over the
  split (real outputs, not hand-picked), ready for a README.
