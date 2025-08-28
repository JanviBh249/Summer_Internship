
**Summer Internship Project – Geospatial Image Classification using Deep Learning**  

---

## 📌 **Project Overview**  
This project was completed during my **Summer Internship (AI Capstone with Deep Learning – Coursera, IBM)**.  
It focuses on **classifying geospatial images (agricultural vs. non-agricultural land)** using advanced **deep learning approaches**:  

- **Convolutional Neural Networks (CNNs)**  
- **Vision Transformers (ViTs)**  
- **Hybrid CNN–ViT Architectures**  

**Guided by:**  
- **Internal Guide**: Prof. Nishant Koshti  

---

## 🛠 **Tools & Technologies**  
- **Frameworks:** TensorFlow/Keras, PyTorch  
- **Platforms:** Google Colab, Jupyter Notebook  
- **Libraries:** torchvision, NumPy, Matplotlib, PIL  
- **Techniques:** Data Augmentation, Transfer Learning  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC  

---

## 📚 **Module-wise Learning**  

### 🔹 **Module 1 – Data Handling & Augmentation**  
- Learned **memory-based vs generator-based pipelines**.  
- Implemented **ImageDataGenerator (Keras)** and **DataLoader + transforms (PyTorch)**.  
- Applied augmentations: rotation, flips, zoom, normalization → improved generalization.  

---

### 🔹 **Module 2 – CNN Development**  
- Built **custom CNNs** in both Keras & PyTorch for farmland classification.  
- Tuned **hyperparameters, optimization strategies, and regularization**.  
- Achieved accuracy up to **~99.8% (PyTorch CNN)**.  

---

### 🔹 **Module 3 – Vision Transformers (ViTs)**  
- Implemented **ViT-B16 with pretrained ImageNet weights (transfer learning)**.  
- Fine-tuned ViTs in both Keras & PyTorch.  
- Achieved accuracy of **~99.6%**.  

---

### 🔹 **Module 4 – Hybrid CNN–ViT Models**  
- Combined **CNN feature extraction** (local patterns) with **ViT attention** (global context).  
- Delivered the **best accuracy (~99.9%)** with **perfect ROC-AUC (1.0)**.  
- Learned **trade-offs**: CNNs (fast), ViTs (contextual), Hybrids (balanced).  

---


---

**requirements.txt** (sample):

```text
tensorflow
torch
torchvision
numpy
matplotlib
pillow
scikit-learn
```

### 4️⃣ Training & Evaluation

* **CNN notebooks:** `cnn_keras.ipynb`, `cnn_pytorch.ipynb`
* **ViT notebooks:** `vit_keras.ipynb`, `vit_pytorch.ipynb`
* **Hybrid model notebook:** `hybrid_model.ipynb`
* Results (ROC curves, confusion matrices) auto-saved in `/results`.

---

## 📖 **Key Takeaways**

* Mastered **data handling, augmentation, CNNs, and Transformers**.
* Learned to compare **frameworks (Keras vs PyTorch)** and **architectures (CNN vs ViT vs Hybrid)**.
* Understood **trade-offs between accuracy, inference speed, and generalization**.
* Developed a **complete end-to-end geospatial deep learning pipeline** during internship.

---

 *This repository reflects the work carried out during my **Summer Internship**, bridging academic learning with real-world **AI applications in geospatial image analysis.***

```



