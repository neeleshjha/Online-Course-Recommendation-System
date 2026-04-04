# 📚 Online Course Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents

- [Business Objective](#-business-objective)
- [Problem Statement](#-problem-statement)
- [Solution Approach](#-solution-approach)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models & Evaluation](#-models--evaluation)
- [Key Outcomes](#-key-outcomes)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)

---

## 🎯 Business Objective

Build an intelligent **online course recommendation engine** that suggests the most relevant courses to learners based on their interests, past enrollments, and engagement levels — reducing information overload on e-learning platforms and improving learner satisfaction, retention, and course completion rates.

---

## ❓ Problem Statement

With thousands of courses available across e-learning platforms (Coursera, Udemy, edX, etc.), learners face significant **choice overload**. The core challenges are:

- Users cannot easily discover courses matching their skill level, learning history, or preferences.
- Platforms lack personalised pathways that adapt to individual user behaviour.
- No single filter (e.g., price, rating, difficulty) alone captures the full picture of relevance.

The goal is to build a **multi-model recommendation system** that addresses these gaps using collaborative filtering, content-based filtering, and a hybrid approach.

---

## 💡 Solution Approach

A **5-model recommendation architecture** was designed, covering different angles of personalisation:

| # | Model | Approach | Input |
|---|---|---|---|
| 1 | **KNN on PCA Features** | Content-Based Filtering | Course feature vectors reduced via PCA |
| 2 | **User-Based KNN** | Collaborative Filtering | User-Item rating matrix |
| 3 | **SVD Matrix Factorization** | Collaborative Filtering | Latent user/course factor decomposition |
| 4 | **Hybrid (CF + CBF)** | Hybrid Filtering | Weighted blend of Models 1 & 2 (α=0.7) |
| 5 | **TF-IDF + Cosine Similarity** | Content-Based (NLP) | Course name, difficulty, certification text |

---

## 📊 Dataset

| Property | Details |
|---|---|
| **File** | `online_course_recommendation_v2.csv` |
| **Rows** | 1,00,000 records |
| **Columns** | 14 features |
| **Users** | Multiple unique learners (`user_id`) |
| **Courses** | Multiple unique courses (`course_id`) |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `user_id` | Integer | Unique learner identifier |
| `course_id` | Integer | Unique course identifier |
| `course_name` | String | Name of the course |
| `instructor` | String | Instructor name |
| `course_duration_hours` | Float (5–100) | Course length in hours |
| `certification_offered` | Yes/No | Whether a certificate is provided |
| `difficulty_level` | Beginner / Intermediate / Advanced | Course difficulty |
| `rating` | Float (1–5) | User-provided course rating |
| `enrollment_numbers` | Integer | Total enrolled students |
| `course_price` | Float (20–500) | Course price (USD) |
| `feedback_score` | Float (0–1) | Normalised student feedback sentiment |
| `study_material_available` | Yes/No | Availability of extra study materials |
| `time_spent_hours` | Float (1–100) | Avg. time spent by students |
| `previous_courses_taken` | Integer | Learner's prior course count |

---

## 🔄 Project Workflow

```
Raw Data (100K rows × 14 cols)
        │
        ▼
    EDA
    ├── Univariate Analysis (distributions, skewness, kurtosis)
    ├── Bivariate Analysis (correlation heatmap, scatter plots, boxplots)
    └── Multivariate Analysis (pair plots, group stats by difficulty level)
        │
        ▼
    Outlier Detection & Treatment
    ├── IQR Method  ── detection
    ├── Z-Score Method  ── detection
    └── IQR Capping  ── treatment
        │
        ▼
    Preprocessing & Feature Engineering
    ├── Label Encoding (binary cats: certification, study_material)
    ├── Ordinal Encoding (difficulty: Beginner→Intermediate→Advanced)
    ├── StandardScaler + MinMaxScaler
    └── 7 Engineered Features
            ├── engagement_score = (time_spent / duration) × rating
            ├── price_per_hour = price / duration
            ├── popularity_score = normalised enrollment
            ├── quality_indicator = rating + feedback_score / 2
            ├── completion_ratio = time_spent / duration
            ├── experience_level = binned previous_courses_taken
            └── value_for_money = rating / (price + 1)
        │
        ▼
    Dimensionality Reduction — PCA
    ├── Scree plot & cumulative variance curve
    └── Retain components explaining 95% variance
        │
        ▼
    Clustering — K-Means on PCA features
    └── Elbow method selects optimal K
        │
        ▼
    5 Recommendation Models
    ├── Model 1: Content-Based KNN (PCA + cosine)
    ├── Model 2: User-Based CF (KNN on user-item matrix)
    ├── Model 3: SVD Matrix Factorization (20 latent factors)
    ├── Model 4: Hybrid CF + CBF (weighted blend)
    └── Model 5: TF-IDF + Cosine Similarity (text features)
        │
        ▼
    Streamlit Deployment
```

---

## 🤖 Models & Evaluation

### Model 1 — Content-Based KNN (PCA)
- `NearestNeighbors(metric='cosine')` on PCA-reduced course feature vectors.
- Returns top-N most similar courses by cosine distance for a given course ID.

### Model 2 — User-Based Collaborative Filtering
- Builds a **User × Course rating pivot table**.
- Finds K most similar users by cosine similarity; recommends their highly-rated unseen courses.

### Model 3 — SVD Matrix Factorization
- `TruncatedSVD(n_components=20)` decomposes the user-item matrix into latent factors.
- Reconstructs predicted ratings for all user-course pairs; recommends highest-predicted unseen courses.

### Model 4 — Hybrid Recommender
- Blends CBF (Model 1) and CF (Model 2) scores with tunable `alpha` weight (`α=0.7` for CF).
- Balances personalised community behaviour with content similarity.

### Model 5 — TF-IDF Content-Based (NLP)
- Constructs text features: `course_name + difficulty_level + certification_offered + experience_level`.
- `TfidfVectorizer(max_features=2000)` + `linear_kernel` (cosine similarity).
- Ideal for **cold-start** scenarios (new users with no rating history).

### Evaluation Metrics

| Metric | Usage |
|---|---|
| Cosine Similarity Score | Content proximity between courses (Models 1, 5) |
| Predicted Rating Score | SVD reconstruction quality (Model 3) |
| User-Item Matrix Sparsity | Data coverage for collaborative models |
| Inertia / Elbow Curve | Optimal cluster count for K-Means |

---

## 📈 Key Outcomes

- Developed a **5-model recommendation system** spanning content-based, collaborative, hybrid, and NLP paradigms.
- Engineered **7 new features** (engagement score, value for money, quality indicator, etc.) enriching model signal.
- PCA retained **95% variance** while significantly reducing input dimensionality for faster, more stable similarity computations.
- SVD factorisation successfully decomposed a 100K-row sparse user-item matrix into 20 latent factors.
- The **Hybrid model** (α=0.7) provided the best practical recommendations by combining content and community signals.
- TF-IDF model solved the **cold-start problem**, enabling recommendations for brand-new users.
- Zero missing values and zero duplicate rows in the dataset — clean baseline confirmed during EDA.

---

## 📁 Project Structure

```
online-course-recommendation/
│
├── data/
│   └── online_course_recommendation_v2.csv           # Raw dataset
│
├── notebooks/
│   └── online_course_recommendation_system.ipynb     # Full EDA + Modelling notebook
│
├── app/
│   └── app.py                                        # Streamlit deployment app
│
├── models/
│   ├── knn_model.pkl                                 # Content-Based KNN
│   ├── user_knn.pkl                                  # User-Based KNN
│   ├── svd_model.pkl                                 # SVD model
│   ├── pca_model.pkl                                 # PCA transformer
│   └── tfidf_vectorizer.pkl                          # TF-IDF vectorizer
│
├── docs/
│   └── Online_Course_Recommendation_Dataset_V2_Documentation.docx
│
├── requirements.txt                                  # Python dependencies
├── .gitignore                                        # Files excluded from version control
├── LICENSE                                           # MIT License
└── README.md                                         # Project documentation (this file)
```

---

## 🛠 Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas, NumPy | Data manipulation & feature engineering |
| Matplotlib, Seaborn | EDA visualisation |
| Scikit-learn | KNN, SVD, PCA, scaling, encoding, TF-IDF |
| SciPy | Statistical analysis (Z-score, chi-square) |
| Pickle | Model serialisation |
| Streamlit | Web application deployment |
| Jupyter Notebook | Development environment |
