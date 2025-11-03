# 🧠 ML-Lab

**ML-Lab** is a collaborative Python-based Machine Learning repository by **Ajith Kumara** and **Dinusha Chamindi Gunathilaka**.  

This project serves as a **hands-on learning laboratory** where each ML concept — from classical algorithms to deep learning — is implemented **from scratch using Python** and paired with clear documentation for deeper understanding.

---

## 🎯 Objectives

- Strengthen the **foundational understanding** of Machine Learning algorithms  
- Build each algorithm **step-by-step using core Python (no shortcuts)**  
- Maintain structured **code + documentation** for easy reference  
- Create a collaborative environment for ongoing ML exploration and improvement  

---

## 🧩 Project Structure
  ---

    ml-lab/
    │
    ├── README.md # Project overview (this file)
    ├── requirements.txt # Python dependencies
    ├── main.py # Optional entry point
    │
    ├── ml_basics/ # Core ML algorithms
    │ ├── init.py
    │ ├── linear_regression.py
    │ ├── logistic_regression.py
    │ ├── decision_tree.py
    │ ├── svm_classifier.py
    │ └── knn_classifier.py
    │
    │
    ├── supervised_learning/ # Clustering and dimensionality reduction
    │  ├── Regression (Predict Continuous Values)
    │  │ ├─ Linear Regression
    │  │ ├─ Polynomial Regression
    │  │ ├─ Ridge Regression
    │  │ └─ Lasso Regression
    │  │  📘 Used for predicting health costs, BMI-based risk scores, etc.
    │  │  ✅ Pros: Interpretable, fast, good baseline
    │  │  ⚠️ Cons: Sensitive to outliers and multicollinearity
    │  │  🎯 Best for: Predicting numeric outcomes
    │  │
    │  ├── Classification (Predict Categories)
    │  │ ├─ Logistic Regression
    │  │ ├─ Decision Trees
    │  │ ├─ Random Forest
    │  │ ├─ Support Vector Machines (SVM)
    │  │ └─ K-Nearest Neighbors (KNN)
    │  │  📘 Used for predicting disease presence, smoker vs. non-smoker, etc.
    │  │  ✅ Pros: Handles categorical data well, powerful ensembles
    │  │  ⚠️ Cons: Overfitting possible, needs parameter tuning
    │  │  🎯 Best for: Health condition classification or diagnosis prediction
    │
    │
    │
    ├── 2. Unsupervised Learning
    │  ├── Clustering (Group similar data points)
    │  │ ├─ K-Means
    │  │ ├─ Hierarchical Clustering
    │  │ └─ DBSCAN
    │  │  📘 Used for grouping patients with similar symptoms or medical histories.
    │  │  ✅ Pros: Reveals hidden structure
    │  │  ⚠️ Cons: No clear accuracy metric
    │  │  🎯 Best for: Patient segmentation, gene clustering
    │  │
    │  ├── Dimensionality Reduction
    │  │ ├─ PCA (Principal Component Analysis)
    │  │ ├─ t-SNE
    │  │ └─ Autoencoders
    │  │  📘 Used for simplifying large health datasets while keeping key info.
    │  │  ✅ Pros: Reduces noise, faster training
    │  │  ⚠️ Cons: May lose interpretability
    │  │  🎯 Best for: Visualization and preprocessing high-dimensional data
    │
    ├── 3. Reinforcement Learning
    │  ├── Value-Based Methods
    │  │ ├─ Q-Learning
    │  │ └─ Deep Q-Networks (DQN)
    │  │  📘 Used for decision-making (e.g., personalized medicine or treatment paths).
    │  │  ✅ Pros: Learns from interaction
    │  │  ⚠️ Cons: Requires lots of training
    │  │  🎯 Best for: Sequential decision optimization
    │  │
    │  ├── Policy-Based Methods
    │  │ ├─ REINFORCE
    │  │ └─ Actor-Critic
    │  │  📘 Used in advanced control and adaptive systems (e.g., robotic surgery).
    │  │  ✅ Pros: Can learn complex policies
    │  │  ⚠️ Cons: Harder to train, unstable sometimes
    │  │  🎯 Best for: Adaptive, real-time optimization problems  
    │
    ├── 4. Reinforcement Learning
    │  ├── Value-Based Methods
    │  │ ├─ Q-Learning
    │  │ └─ Deep Q-Networks (DQN)
    │  │  📘 Used for decision-making (e.g., personalized medicine or treatment paths).
    │  │  ✅ Pros: Learns from interaction
    │  │  ⚠️ Cons: Requires lots of training
    │  │  🎯 Best for: Sequential decision optimization
    │  │
    │  ├── Policy-Based Methods
    │  │ ├─ REINFORCE
    │  │ └─ Actor-Critic
    │  │  📘 Used in advanced control and adaptive systems (e.g., robotic surgery).
    │  │  ✅ Pros: Can learn complex policies
    │  │  ⚠️ Cons: Harder to train, unstable sometimes
    │  │  🎯 Best for: Adaptive, real-time optimization problems  
    │

    ├── unsupervised_learning/ # Clustering and dimensionality reduction
    │ ├── init.py
    │ ├── kmeans.py
    │ ├── hierarchical_clustering.py
    │ └── pca.py
    │
    ├── deep_learning/ # Neural networks and advanced topics
    │ ├── init.py
    │ ├── neural_network_basics.py
    │ ├── cnn_example.py
    │ └── rnn_example.py
    │
    ├── utils/ # Helper modules
    │ ├── init.py
    │ ├── data_preprocessing.py
    │ ├── model_evaluation.py
    │ └── visualization.py
    │
    └── docs/ # Documentation and theory notes
    ├── linear_regression.md
    ├── kmeans.md
    ├── cnn_basics.md
    └── overview.md

   ---

## ⚙️ Installation

1. Clone the repository  
   ```
   bash
   git clone https://github.com/<your-username>/ml-lab.git
   cd ml-lab
   ```
2. (Optional) Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

   ```

4. Install dependencies
   ```
   pip install -r requirements.txt

   ```
