# Multiclass Wine Quality Classification & Diagnostic Analysis

An empirical machine learning study developed for **CS 178 (Machine Learning)** at UC Irvine, evaluating distance-based, linear, nonlinear, and tree-based models on multiclass physicochemical wine data[cite: 13, 14, 15].

---

## Problem Formulation & Dataset

* **Objective:** Predict discrete wine quality ratings (ranging from 3 to 9) using measured chemical attributes[cite: 14, 15].
* **Dataset:** Combined red (1,599 samples) and white (4,898 samples) wine datasets totaling **6,497 samples** across 11 physicochemical features[cite: 14].
* **Task Type:** Multiclass classification[cite: 14, 15].
* **Class Imbalance:** Most samples concentrate around ratings 5 and 6, while extreme ratings (3, 4, 8, 9) are sparse[cite: 14].

---

## Exploratory Data Analysis & Visualizations

<!-- 
  TODO: Save your combined distribution plot into 'assets/wine_distributions.png' 
  and your diagnostic plots into 'assets/' (or adjust paths to match your folder).
-->

### 1. Wine Quality & Type Distributions

![Wine Quality Distributions](dist_quality_type.png)

* **Overall Distribution (Left):** Severe class imbalance across the dataset, with sample counts heavily concentrated around ratings 5 and 6, while extreme quality scores (3, 4, 8, and 9) are sparse[cite: 14].
* **Quality by Wine Type (Right):** Breakdown showing white wine samples dominate total volume, with both red and white varieties peaking at quality scores 5 and 6[cite: 14].

---

### 2. Model Diagnostics & Error Analysis

| Decision Tree Complexity (Depth vs. Accuracy) | Confusion Matrix Analysis |
| :---: | :---: |
| ![Depth vs Accuracy Plot](assets/tree_depth_curve.png) | ![Confusion Matrix](assets/confusion_matrix.png) |

---

## Classifiers & Hyperparameter Optimization

Models were implemented in Python using **scikit-learn**, with preprocessing using `StandardScaler` (fit strictly on training data) and evaluated using an 80/20 stratified train/test split[cite: 15].

| Classifier | Type | Hyperparameter Search Space | Best Configuration |
| :--- | :--- | :--- | :--- |
| **$k$-Nearest Neighbors ($k\text{NN}$)**[cite: 14] | Non-parametric instance-based[cite: 14] | $k \in \{1, 3, 5, \dots, 29\}$ (`weights='uniform'`)[cite: 14] | $k = 1$[cite: 16] |
| **Logistic Regression**[cite: 14, 15] | Linear baseline (L-BFGS)[cite: 14, 15] | $C \in \{0.001, 0.01, 0.1, 1, 10, 100\}$, $\text{max\_iter} = 1000$[cite: 15] | Tuned via CV[cite: 15] |
| **Neural Network (MLP)**[cite: 14, 15] | Nonlinear feedforward[cite: 14, 15] | Topologies: `(50,)`, `(100,)`, `(50,25)`, `(100,50)`, `(100,50,25)`; ReLU, Adam, $\alpha=0.001$[cite: 15] | Tuned via CV[cite: 15] |
| **Decision Tree**[cite: 14, 15] | Recursive space partitioning[cite: 15] | $\text{max\_depth} \in \{3, 5, \dots, 20, \text{None}\}$, $\text{min\_samples\_split}$, $\text{min\_samples\_leaf}$, $\text{max\_features}$[cite: 15] | Grid search tuned[cite: 15] |

---

## Experimental Results

Final model performance evaluated on the held-out 20% test partition[cite: 15, 16]:

| Model | Best CV / Validation Accuracy | Test Accuracy | Observations & Key Metrics[cite: 16] |
| :--- | :---: | :---: | :--- |
| **$k$-Nearest Neighbors ($k\text{NN}$)**[cite: 16] | **0.5948**[cite: 16] | **0.6477**[cite: 16] | Highest test accuracy; weighted precision, recall, and F1 $\approx 0.65$[cite: 16]. |
| **Decision Tree**[cite: 16] | 0.5896[cite: 16] | 0.6162[cite: 16] | High training accuracy (0.9894 tuned, 1.0000 untuned) showing strong overfitting[cite: 16]. |
| **Neural Network (MLP)**[cite: 16] | 0.5605[cite: 16] | 0.5692[cite: 16] | Weighted F1 $\approx 0.53$; outperformed linear model but constrained by sample size[cite: 16, 17]. |
| **Logistic Regression**[cite: 16] | 0.5417[cite: 16] | 0.5308[cite: 16] | Baseline linear model; weighted F1 $\approx 0.48$[cite: 16]. |

---

## Key Engineering & Scientific Insights

* **Accuracy Obscures Per-Class Failure:** While overall test accuracies reached up to $\approx 65\%$, confusion matrices proved that models predominantly succeeded on modal classes (5 and 6) and struggled heavily on boundary classes (3, 4, 8, 9) due to extreme class imbalance[cite: 16].
* **Adjacent Error Concentration:** Errors primarily occurred between neighboring ratings (e.g., misclassifying 5 as 6, or 6 as 7), indicating continuous underlying chemical traits that make strict discrete boundaries difficult to separate[cite: 16, 17].
* **Top Predictive Features:** Feature importance analysis from the Decision Tree identified **alcohol**, **volatile acidity**, and **free sulfur dioxide** as the most influential predictors of wine quality[cite: 16].
* **Flexibility vs. Overfitting:** Flexible, non-linear classifiers consistently outperformed the linear baseline, but models like Decision Trees required strict regularization to prevent complete memorization of training instances[cite: 16, 17].

---

## Authors & Team Contributions

* **Leo Lin:** Implemented and tuned classifiers ($k\text{NN}$ and Logistic Regression), conducted cross-validation experiments, organized performance results, and authored the Classifiers & Experimental Results sections[cite: 13, 17].
* **Natalie Yang:** Implemented Neural Network and Decision Tree experiments, analyzed overfitting dynamics, confusion matrices, and feature importance[cite: 13, 17].
* **Anjali Janavi Alwar:** Handled dataset preprocessing, merging red/white corpora, exploratory data visualization, and authored the Data Description[cite: 13, 17].
* **Kelly Wu:** Compared evaluation metrics across models, interpreted error trends and confusion matrices, and compiled the final academic report[cite: 13, 17].
