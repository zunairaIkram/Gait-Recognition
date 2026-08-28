# Human Gait Recognition by Silhouette Extraction

Appearance-based, model-free gait recognition. Walking frame sequences are converted to silhouettes (pre-trained HumanMatting), averaged into Gait Energy Images (GEIs), and classified with PCA + Linear SVM.

## Pipeline

```
Raw frames → Silhouette extraction (HumanMatting / ResNet50) → GEI → PCA + LinearSVC → Person ID
```

| Step | Output |
|------|--------|
| Load Data | `image_data.pkl` |
| Silhouette Extraction | `silhouette/` images, `silhouette_data.pkl` |
| GEI Extraction | `gei/` images, `gei_data.pkl` |
| Train Model | `finalized_model_11_labels.sav`, `pca_model_11_labels.sav` |
| Predict | Person label per test GEI image |

## Requirements

- Python 3.8+
- PyTorch, torchvision
- OpenCV, Pillow, NumPy, scikit-learn, scikit-image, imageio
- matplotlib, customtkinter

```bash
pip install torch torchvision opencv-python pillow numpy scikit-learn scikit-image imageio matplotlib customtkinter tqdm pandas seaborn
```

GPU optional; CUDA is used automatically when available.

## Dataset Layout

Organize training images as `person_id / scene_id / frames`:

```
images/
├── 001/
│   ├── scene1/
│   │   ├── frame001.jpg
│   │   └── frame002.jpg
│   └── scene2/
│       └── ...
├── 002/
│   └── ...
```

- `001`, `002`, … — class labels (person IDs)
- `scene1`, `scene2`, … — one walking sequence each (many frames → one GEI)

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

## Configuration

Update hardcoded paths before running:

**`src/main.py`** — data and artifact locations:

```python
data_file = r".../image_data.pkl"
silhouette_data_file = r".../silhouette_data.pkl"
silhouette_folder = r".../silhouette"
gei_data_file = r".../gei_data.pkl"
gei_folder = r".../gei"
```

**`src/silhouette_extraction.py`** — pre-trained weights:

```python
pretrained_weight = r".../src/pretrained/SGHM-ResNet50.pth"
```

Place `SGHM-ResNet50.pth` in `src/pretrained/`. Weights are not included in this repo.

## Run

From the `src` directory:

```bash
cd src
python GUI.py
```

### Tab 1 — Training (run in order)

1. **Enter dataset path** → `images/` root (person folders).
2. **Load Data** — reads frames, caches to `image_data.pkl`.
3. **Silhouette Extraction** — HumanMatting per frame; saves binary silhouettes.
4. **GEI Extraction** — averages silhouettes per scene; crops/resizes to 128×64.
5. **Train Model** — 70/30 split, PCA (99% variance), LinearSVC with C tuning; saves `.sav` models.

Each step requires the previous step’s output. Existing pickle files are reused (steps are skipped if cache exists).

### Tab 2 — Prediction

1. Select test **GEI-format** grayscale images (same preprocessing as training GEIs).
2. **Predict Gait** — loads saved PCA + SVM; shows original vs predicted label.

Prediction expects GEI images, not raw walking frames.
