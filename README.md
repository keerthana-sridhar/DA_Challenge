# # Kaggle Data Challenge: Model Approaches and Results

This repository contains all experiments performed for predicting quality scores for model-generated responses using text–metric alignment. Multiple modeling strategies were explored, and each notebook corresponds to a specific approach. The generated submission CSV files for every approach are also included.

---

## 1. Overview of the Task

The goal is to predict a score (0–10) for a response given:
- system prompt  
- user prompt  
- model response  

The dataset includes:
- `train_data.json`, `test_data.json`  
- `metric_name_embeddings.npy`  
- `metric_names.json`  

Text and metric embeddings were combined to build several regression and contrastive models.

---

## 2. Approaches Implemented

### 2.1 Classical Regression Models
Notebooks:  
- `data_initial_lightgbm.ipynb`  
- `lightgbm2.ipynb`  

Methods:
- Linear Regression  
- Ridge Regression  
- LightGBM Regressor with tuned hyperparameters  
- Additional engineered distance and similarity features  

Output files:  
- `lightgbm_predictions2 (1).csv`


---

## 3. Approach 2: Contrastive Learning with Negative Sampling
Notebooks:  
- `mlp1.ipynb`
- `mlp_and_dual_encoder.ipynb`

Key ideas:
- Combine text and metric embeddings  
- Generate negative metric samples per training pair  
- Train with contrastive loss, InfoNCE, ranking loss, and regression  
- Use a small MLP on concatenated embeddings  

Output:  
- `contrastive_mlp.csv`
- `contrastive_de.csv`

---

## 4. Approach 3: Dual Encoder + Shared Latent Space
Notebooks:  
- `final.ipynb`  

Highlights:
- Separate projectile networks for text vs metric embeddings  
- L2-normalized latent representations  
- Dot-product similarity + regression head  
- Gentle LayerNorm to stabilize embedding space  
- Combination of contrastive and regression objectives  

Variants explored:
- Dual encoder + two expert regressors  
- Dual encoder + single regression head  
- MPNet, Gemma, and precomputed embeddings  

Output files include:
- `final.csv`  


---
Model:  
- Dual-encoder architecture  
- Gentle LayerNorm  
- Learnable temperature for similarity scaling  
- Margin loss + positive/negative BCE + regression  
- Trained for ~25 epochs  

Performance:  
- Test RMSE: **1.7–1.8**

Final submission:  
- `submission_soft_layernorm.csv`

---

## 6. Repository Contents

