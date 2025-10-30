# ML Playground: Vision, Representation Learning, Clustering & Tabular Modeling

This repository contains five Jupyter notebooks demonstrating practical machine learning and deep learning workflows across **vision (CLIP, image embeddings, zero-shot)**, **unsupervised learning (PCA, t-SNE, clustering)**, and **tabular supervised learning (model selection + stacking)**.

The emphasis is on:
- using well-known ML/DL libraries (**PyTorch**, **OpenAI CLIP**, **scikit-learn**, **pandas**, **matplotlib**),
- applying **dimensionality reduction** to high-dimensional data,
- comparing **classical ML models**,
- and visualizing **semantic structure** in the data.

---

## Notebooks

### 1. `clip_image_embeddings_tsne_visualization.ipynb`
**Purpose:** Extract semantic image representations with **CLIP** and visualize their structure.

**Key points:**
- **Libraries / Tech:** PyTorch, OpenAI CLIP, pandas, NumPy, scikit-learn (t-SNE), matplotlib.
- **Pipeline:**
  1. Load pretrained **CLIP** (image encoder).
  2. Preprocess images into CLIP’s expected format.
  3. Compute **image embeddings** (high-dimensional vector per image).
  4. Apply **t-SNE** to project embeddings to 2D.
  5. Visualize clusters to see if semantically similar images are grouped.
- **What it shows:** ability to work with **pretrained vision-language models**, do **feature extraction** instead of full training, and **interpret latent space**.

---

### 2. `clip_zero_shot_on_extracted_images_tsne.ipynb`
**Purpose:** Run **zero-shot image classification** with CLIP on a folder of images and analyze the embedding space.

**Key points:**
- **Libraries / Tech:** PyTorch, OpenAI CLIP, scikit-learn (t-SNE), matplotlib.
- **Pipeline:**
  1. Load images from disk (unzipped folder).
  2. Define **text prompts / class names**.
  3. Use CLIP to compute **image–text similarities** → zero-shot labels.
  4. Extract image embeddings.
  5. Run **t-SNE** to see whether CLIP’s predicted classes form coherent regions.
- **What it shows:** familiarity with **zero-shot learning**, **prompt-based classification**, and **qualitative embedding evaluation** (visualizing if predicted labels make sense).

---

### 3. `star_classification_model_selection_stacking.ipynb`
**Purpose:** Supervised learning on a **tabular astronomy / star classification** dataset with **model comparison** and **stacking**.

**Key points:**
- **Libraries / Tech:** pandas, NumPy, **scikit-learn** (train/test split, preprocessing, classifiers, metrics), possibly `LabelEncoder`/`OneHotEncoder`.
- **Models compared (typical in this notebook):**
  - Logistic Regression
  - Random Forest / Tree-based model
  - Support Vector Machine / Linear SVC
  - k-NN or Gradient Boosting (depending on the cells)
  - **Stacking / ensemble** to combine best base learners
- **Data handling:**
  - Load `StarClassificationDataset.csv`
  - Basic cleaning + feature/target split
  - Encode categorical features
- **Evaluation metrics:**
  - **Accuracy**
  - **Precision / Recall / F1** (macro/micro depending on class balance)
  - **Confusion matrix** for per-class performance
- **What it shows:** ability to do **structured ML experiments**, run **multiple baseline models**, and **justify a final model via metrics** — exactly what you’d do in an ML assignment or POC.

---

### 4. `rocks360_dim_reduction_pca_tsne_partial.ipynb`
**Purpose:** Explore an image/perceptual dataset (e.g. 360 rock images + human MDS data) using **unsupervised representation learning**.

**Key points:**
- **Libraries / Tech:** pandas/NumPy, **scikit-learn** (PCA, t-SNE), matplotlib.
- **Pipeline:**
  1. Load image-derived features (or flattened image vectors).
  2. Apply **PCA** to reduce dimensionality (variance-preserving linear projection).
  3. Apply **t-SNE** to capture non-linear structure for visualization.
  4. (In the fuller version, compare to **human MDS** to see how close the machine representation is to human similarity judgments.)
- **Status:** **partial / in-progress** → still useful to show DR workflow.
- **What it shows:** understanding of **when to use PCA vs. t-SNE**, how to **prepare high-dimensional data**, and how to **interpret clustering structure** without labels.

---

### 5. `image_clustering_pca_kmeans_gmm_eval.ipynb`
**Purpose:** Unsupervised **image clustering**: reduce → cluster → evaluate.

**Key points:**
- **Libraries / Tech:** pandas/NumPy, **scikit-learn** (PCA, KMeans, GaussianMixture), matplotlib.
- **Pipeline:**
  1. Preprocess / reshape image data into feature vectors.
  2. **PCA** to reduce to a compact representation (faster clustering, noise reduction).
  3. **KMeans** over several `k` values.
  4. **Gaussian Mixture Models (GMM)** to allow **soft clustering** / non-spherical clusters.
  5. **Evaluation / inspection:**
     - inertia (for KMeans)
     - log-likelihood / BIC/AIC (for GMM if present)
     - sometimes silhouette score
  6. Visualize clusters (possibly in 2D PCA space).
- **What it shows:** hands-on with **unsupervised learning**, ability to **compare clustering algorithms**, and understanding that images often need **feature extraction + DR** before clustering.

---

## Tech Stack (Overall)

- **Core Python / Data:** `pandas`, `numpy`
- **Modeling / ML:** `scikit-learn`
  - supervised: classification, model selection, stacking
  - unsupervised: PCA, t-SNE, KMeans, GaussianMixture
- **Deep Learning / Vision:** `torch`, **OpenAI CLIP**
- **Visualization:** `matplotlib` (and sometimes 2D embedding plots)
- **Tasks covered:**
  - Zero-shot image classification
  - Image embedding extraction
  - Dimensionality reduction (PCA, t-SNE)
  - Unsupervised clustering (KMeans, GMM)
  - Supervised classification on tabular data
  - Model comparison + evaluation

---

## Suggested Repository Name

**`ml-playground-vision-tabular-unsupervised`**

(accurately signals: vision ✅, tabular ✅, unsupervised ✅)
