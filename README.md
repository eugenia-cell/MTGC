# Multi-Scale Temporal Graph Contrastive Embedding for Urban Region Representation (MTGC)

This is the implementation of **Multi-Scale Temporal Graph Contrastive Embedding for Urban Region Representation (MTGC)** in the following paper:


📄 [Link to the paper]()

---

## Table of Contents

- [Data](#data)
- [Requirements](#requirements)
- [QuickStart](#quickstart)
- [Reference](#reference)

---

## 📦 Data

Here we provide the processed data used in our paper.

- 📁 Raw data source: 
[San Francisco Open Data](https://datasf.org/opendata/)
[NYC Open Data](https://opendata.cityofnewyork.us/)

- 📌 Task: Crime Prediction, Check-in Prediction, Landuse Clustering.

---

## 📋 Requirements

```bash
Python >= 3.10
pytorch >= 2.6.0
numpy >= 1.24.3
pandas >= 2.2.3
sklearn >= 1.6.1
```

## 🚀 QuickStart
```bash
cd code
cd manhattan 
(cd SF)
python train.py (CPU)
python gtrain.py (GPU)
```

## 📚 Reference
