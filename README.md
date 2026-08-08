# Plant Disease Classification — From-Scratch CNNs vs. Transfer Learning

Most plant-disease classifiers report one accuracy number and call it a day. This project asks a more honest question: *which* diseases does the model actually get wrong, and *why* — is it the model's fault, or do the diseases just look too similar to tell apart?

To find out, I trained three models on the same data and compared not just their accuracy, but the specific mistakes they make.

---

## TL;DR

- Trained a **CNN from scratch**, fine-tuned a **pretrained ResNet-18**, and ran a **data-augmentation ablation** — all on a balanced 10-class subset of PlantVillage (tomato, corn, grape, potato).
- Transfer learning jumped accuracy from **91.0% → 97.8%**. Augmentation added only ~1 point, and mostly just reduced overfitting.
- **The interesting bit:** tomato early blight vs. late blight stayed the hardest pair in *every* model — the lowest F1 scores across the board, even after transfer learning cleaned up almost everything else. That points to the two diseases genuinely looking alike, not the model being weak.
- Full write-up in [`preprint/Plant_Disease_Classification.pdf`](preprint/Plant_Disease_Classification.pdf).

---

## Why I built this

I grow plants, and one summer I was raising about ten tomato plants. One of them caught a blight I couldn't identify — and by the time I figured out something was wrong, it had spread and taken out most of my crop. That got me wondering whether a model could tell these diseases apart from a photo, and whether some of them are simply too similar to separate reliably. This project is my attempt to check.

---

## What's in this repo

```
├── Plant_Disease_Diagnosis.ipynb   # the full notebook (data → training → evaluation)
├── preprint/
│   ├── Plant_Disease_Classification.pdf                # the paper
├── figures/                        # learning curves + confusion matrices
├── requirements.txt
└── README.md
```

---

## The setup

**Data.** A deliberately balanced 10-class subset of [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) — 600 images per class, four crops. I picked the classes on purpose: balanced sizes so accuracy doesn't lie, multiple crops so the model has to learn both crop *and* disease, and a few visually similar disease pairs (early vs. late blight) so there'd be something real to analyze in the failure cases.

**Split.** 70/15/15 train/validation/test, fixed seed, test set touched exactly once at the end.

**Models.**
1. A small from-scratch CNN (3 conv blocks + 2 dense layers, dropout) — the baseline.
2. ResNet-18 pretrained on ImageNet, fine-tuned on the leaf data — the transfer-learning model.
3. The from-scratch CNN retrained with augmentation — the ablation.

---

## Results

| Model | Test accuracy |
|---|---|
| From-scratch CNN | 91.0% |
| From-scratch CNN + augmentation | 92.0% |
| Fine-tuned ResNet-18 | **97.8%** |

And the finding that actually matters — the tomato blight confusion, which refused to go away:

| Model | Early↔Late blight errors |
|---|---|
| From-scratch | 14 |
| + augmentation | 9 |
| ResNet-18 | 5 |

Even at 97.8% overall, ResNet-18's two weakest classes were *still* the tomato blights. Everything else it nailed. The most reasonable read: those two diseases just look alike in these images, so a real diagnostic tool should flag uncertainty on that pair rather than guess confidently.

---

## How to run it

Open the notebook in [Google Colab](https://colab.research.google.com/) (a GPU runtime makes it quick), or run locally:

```bash
pip install -r requirements.txt
```

The notebook walks through everything end to end: downloading the data, building and training each model, and generating the confusion matrices and metrics. It's commented so you can follow *why* each step is there, not just what it does.

---

## Honest limitations

This is a scoped study, not a production system. It uses a subset rather than the full dataset, so the accuracy numbers aren't directly comparable to full-dataset benchmarks. PlantVillage images have clean backgrounds, so none of this is tested on messy real-world field photos. And the "these diseases are just similar" conclusion is an inference from the models' consistent behavior — properly confirming it would need something like Grad-CAM or a human-expert baseline on the same images.

---

## Author

**Syeda Aliza Ayaz** — syedaalizaayaz.1606@gmail.com

Full methodology and analysis in the [preprint](preprint/Plant_Disease_Classification.pdf).