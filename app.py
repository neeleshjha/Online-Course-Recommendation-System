"""
app.py — Streamlit Deployment App
Online Course Recommendation System
=====================================
Run with:  streamlit run app.py

Models available:
  1. Content-Based KNN (PCA features + cosine similarity)
  2. User-Based Collaborative Filtering (KNN on user-item matrix)
  3. SVD Matrix Factorization
  4. Hybrid (CF + CBF weighted blend)
  5. TF-IDF Content-Based (NLP)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Recommender",
    page_icon="📚",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
NUMERICAL_FEATURES = [
    "course_duration_hours", "rating", "enrollment_numbers",
    "course_price", "feedback_score", "time_spent_hours",
    "previous_courses_taken",
]
ALPHA = 0.7  # hybrid weight: CF contribution


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file) -> dict:
    """
    Full preprocessing pipeline mirroring the notebook:
    - Outlier capping (IQR)
    - Label + Ordinal encoding
    - StandardScaler + MinMaxScaler
    - Feature engineering (7 new features)
    - User / Course level aggregations
    - Interaction features
    Returns a dict of all objects needed by models.
    """
    df = pd.read_csv(uploaded_file)

    df_processed = df.copy()

    # ── Outlier capping (IQR) ─────────────────────────────────────────────────
    for col in NUMERICAL_FEATURES:
        Q1, Q3 = df_processed[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_processed[col] = df_processed[col].clip(
            lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR
        )

    # ── Encoding ──────────────────────────────────────────────────────────────
    label_encoders = {}
    for col in ["certification_offered", "study_material_available"]:
        le = LabelEncoder()
        df_processed[col + "_encoded"] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    oe = OrdinalEncoder(categories=[["Beginner", "Intermediate", "Advanced"]])
    df_processed["difficulty_level_encoded"] = oe.fit_transform(
        df_processed[["difficulty_level"]]
    )

    # ── Scaling ───────────────────────────────────────────────────────────────
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    df_processed[["scaled_" + c for c in NUMERICAL_FEATURES]] = (
        scaler_standard.fit_transform(df_processed[NUMERICAL_FEATURES])
    )
    df_processed[["normalized_" + c for c in NUMERICAL_FEATURES]] = (
        scaler_minmax.fit_transform(df_processed[NUMERICAL_FEATURES])
    )

    # ── Feature Engineering ───────────────────────────────────────────────────
    df_processed["engagement_score"] = (
        df_processed["time_spent_hours"] / df_processed["course_duration_hours"]
    ) * df_processed["rating"]

    df_processed["price_per_hour"] = (
        df_processed["course_price"] / df_processed["course_duration_hours"]
    )

    enroll = df_processed["enrollment_numbers"]
    df_processed["popularity_score"] = (enroll - enroll.min()) / (enroll.max() - enroll.min())

    df_processed["quality_indicator"] = (
        df_processed["rating"] + df_processed["feedback_score"] / 2
    )

    df_processed["completion_ratio"] = (
        df_processed["time_spent_hours"] / df_processed["course_duration_hours"]
    )

    df_processed["experience_level"] = pd.cut(
        df_processed["previous_courses_taken"],
        bins=[-1, 2, 5, 10, 20],
        labels=["Beginner", "Intermediate", "Advanced", "Expert"],
    )

    df_processed["value_for_money"] = df_processed["rating"] / (
        df_processed["course_price"] + 1
    )

    # ── Aggregations ──────────────────────────────────────────────────────────
    user_stats = df_processed.groupby("user_id").agg(
        user_avg_rating=("rating", "mean"),
        user_rating_std=("rating", "std"),
        user_course_count=("rating", "count"),
        user_avg_time=("time_spent_hours", "mean"),
        user_total_time=("time_spent_hours", "sum"),
        user_avg_price=("course_price", "mean"),
    ).reset_index()
    df_processed = df_processed.merge(user_stats, on="user_id", how="left")

    course_stats = df_processed.groupby("course_id").agg(
        course_avg_rating=("rating", "mean"),
        course_rating_count=("rating", "count"),
        course_avg_enrollment=("enrollment_numbers", "mean"),
        course_avg_feedback=("feedback_score", "mean"),
    ).reset_index()
    df_processed = df_processed.merge(course_stats, on="course_id", how="left")

    # ── Interaction Features ──────────────────────────────────────────────────
    df_processed["rating_experience_interaction"] = (
        df_processed["rating"] * df_processed["previous_courses_taken"]
    )
    df_processed["price_certification_interaction"] = (
        df_processed["course_price"] * df_processed["certification_offered_encoded"]
    )
    df_processed["duration_difficulty_interaction"] = (
        df_processed["course_duration_hours"] * (df_processed["difficulty_level_encoded"] + 1)
    )

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_features = [c for c in df_processed.columns if c.startswith("scaled_")]
    X_pca = df_processed[pca_features].values

    pca_full = PCA()
    pca_full.fit(X_pca)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_opt = int(np.argmax(cumvar >= 0.95)) + 1

    pca = PCA(n_components=n_opt)
    X_pca_transformed = pca.fit_transform(X_pca)
    pca_cols = [f"PC{i+1}" for i in range(n_opt)]
    df_pca = pd.DataFrame(X_pca_transformed, columns=pca_cols)
    for col in pca_cols:
        df_processed[col] = df_pca[col].values

    # ── K-Means clustering ────────────────────────────────────────────────────
    X_cluster = df_pca[pca_cols[:3]].values if n_opt >= 3 else df_pca[pca_cols[:2]].values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_processed["cluster"] = kmeans.fit_predict(X_cluster)

    # ── Course Catalog ────────────────────────────────────────────────────────
    course_catalog = (
        df_processed.groupby("course_id")
        .agg(
            course_name=("course_name", "first"),
            certification_offered=("certification_offered", "first"),
            difficulty_level=("difficulty_level", "first"),
            rating=("rating", "mean"),
            course_price=("course_price", "mean"),
            feedback_score=("feedback_score", "mean"),
            enrollment_numbers=("enrollment_numbers", "mean"),
        )
        .reset_index()
    )

    # ── User-Item Matrix ──────────────────────────────────────────────────────
    user_item_matrix = df_processed.pivot_table(
        index="user_id", columns="course_id", values="rating", fill_value=0
    )

    # ── Model 1: Content-Based KNN ────────────────────────────────────────────
    X_courses = df_pca.values
    knn_model = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
    knn_model.fit(X_courses)
    course_index_map = df_processed[["course_id"]].reset_index(drop=True)

    # ── Model 2: User-Based KNN ───────────────────────────────────────────────
    user_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
    user_knn.fit(user_item_matrix)

    # ── Model 3: SVD ──────────────────────────────────────────────────────────
    svd_model = TruncatedSVD(n_components=20, random_state=42)
    user_features = svd_model.fit_transform(user_item_matrix)
    predicted_ratings = np.dot(user_features, svd_model.components_)
    predicted_ratings_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns,
    )

    # ── Model 5: TF-IDF ───────────────────────────────────────────────────────
    course_df = (
        df_processed.drop_duplicates(subset=["course_name"])[
            ["course_id", "course_name", "difficulty_level",
             "certification_offered", "rating", "experience_level"]
        ]
        .reset_index(drop=True)
    )
    course_df["course_text"] = (
        course_df["course_name"].astype(str) + " " +
        course_df["difficulty_level"].astype(str) + " " +
        course_df["certification_offered"].astype(str) + " " +
        course_df["experience_level"].astype(str)
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf_matrix = tfidf.fit_transform(course_df["course_text"])
    tfidf_course_index = pd.Series(course_df.index, index=course_df["course_name"])

    return {
        "df": df,
        "df_processed": df_processed,
        "df_pca": df_pca,
        "pca": pca,
        "course_catalog": course_catalog,
        "user_item_matrix": user_item_matrix,
        "knn_model": knn_model,
        "X_courses": X_courses,
        "course_index_map": course_index_map,
        "user_knn": user_knn,
        "predicted_ratings_df": predicted_ratings_df,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_course_index": tfidf_course_index,
        "course_df": course_df,
        "pca_full": pca_full,
        "kmeans": kmeans,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def recommend_knn(course_id: int, top_n: int, state: dict) -> pd.DataFrame:
    """Model 1: Content-Based KNN on PCA features."""
    cmap = state["course_index_map"]
    if course_id not in cmap["course_id"].values:
        return pd.DataFrame()
    idx = cmap[cmap["course_id"] == course_id].index[0]
    distances, indices = state["knn_model"].kneighbors(
        state["X_courses"][idx].reshape(1, -1), n_neighbors=top_n + 1
    )
    sim_indices = indices[0][1:]
    sim_dists = distances[0][1:]
    rec_ids = cmap.iloc[sim_indices]["course_id"].values
    result = state["course_catalog"][state["course_catalog"]["course_id"].isin(rec_ids)].copy()
    sim_map = {cmap.iloc[i]["course_id"]: round(1 - d, 4)
               for i, d in zip(sim_indices, sim_dists)}
    result["cosine_similarity"] = result["course_id"].map(sim_map)
    result["rank"] = result["course_id"].apply(lambda x: list(rec_ids).index(x))
    return result.sort_values("rank").drop(columns="rank").reset_index(drop=True)


def recommend_user_based(user_id: int, top_n: int, state: dict) -> pd.DataFrame:
    """Model 2: User-Based Collaborative Filtering."""
    uim = state["user_item_matrix"]
    if user_id not in uim.index:
        return pd.DataFrame()
    user_vec = uim.loc[user_id].values.reshape(1, -1)
    distances, indices = state["user_knn"].kneighbors(user_vec)
    similar_users = uim.iloc[indices[0][1:]]
    mean_scores = similar_users.mean(axis=0)
    already_seen = uim.loc[user_id] > 0
    recs = mean_scores[~already_seen].sort_values(ascending=False).head(top_n)
    result = state["course_catalog"][state["course_catalog"]["course_id"].isin(recs.index)].copy()
    result["predicted_score"] = result["course_id"].map(recs.to_dict())
    return result.sort_values("predicted_score", ascending=False).reset_index(drop=True)


def recommend_svd(user_id: int, top_n: int, state: dict) -> pd.DataFrame:
    """Model 3: SVD Matrix Factorization."""
    prd = state["predicted_ratings_df"]
    uim = state["user_item_matrix"]
    if user_id not in prd.index:
        return pd.DataFrame()
    user_preds = prd.loc[user_id]
    already_taken = uim.loc[user_id] > 0
    recs = user_preds[~already_taken].sort_values(ascending=False).head(top_n)
    result = state["course_catalog"][state["course_catalog"]["course_id"].isin(recs.index)].copy()
    result["predicted_rating"] = result["course_id"].map(recs.to_dict())
    return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)


def recommend_hybrid(user_id: int, top_n: int, alpha: float, state: dict) -> pd.DataFrame:
    """Model 4: Hybrid CF + CBF."""
    uim = state["user_item_matrix"]
    cmap = state["course_index_map"]
    X = state["X_courses"]
    knn = state["user_knn"]
    catalog = state["course_catalog"]

    if user_id not in uim.index:
        return pd.DataFrame()

    # CF scores
    user_vec = uim.loc[user_id].values.reshape(1, -1)
    distances, indices = knn.kneighbors(user_vec)
    similar_users = uim.iloc[indices[0][1:]]
    cf_scores = similar_users.mean(axis=0)

    # CBF scores
    rated = uim.loc[user_id][uim.loc[user_id] > 0].index.tolist()
    cbf_scores_dict: dict = {}
    for cid in rated:
        if cid not in cmap["course_id"].values:
            continue
        idx = cmap[cmap["course_id"] == cid].index[0]
        dists, inds = state["knn_model"].kneighbors(X[idx].reshape(1, -1), n_neighbors=11)
        for d, i in zip(dists[0][1:], inds[0][1:]):
            ncid = cmap.iloc[i]["course_id"]
            cbf_scores_dict[ncid] = cbf_scores_dict.get(ncid, 0) + (1 - d)

    cbf_series = pd.Series(cbf_scores_dict)
    if len(cbf_series) > 0:
        cbf_series = (cbf_series - cbf_series.min()) / (cbf_series.max() - cbf_series.min() + 1e-9)
    if len(cf_scores) > 0:
        cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)
    else:
        cf_norm = cf_scores

    all_ids = set(cf_norm.index) | set(cbf_series.index)
    hybrid: dict = {}
    for cid in all_ids:
        cf_val = float(cf_norm.get(cid, 0))
        cbf_val = float(cbf_series.get(cid, 0))
        hybrid[cid] = alpha * cf_val + (1 - alpha) * cbf_val

    already_seen = set(uim.loc[user_id][uim.loc[user_id] > 0].index)
    hybrid_filtered = {k: v for k, v in hybrid.items() if k not in already_seen}
    top_ids = sorted(hybrid_filtered, key=hybrid_filtered.get, reverse=True)[:top_n]

    result = catalog[catalog["course_id"].isin(top_ids)].copy()
    result["hybrid_score"] = result["course_id"].map(hybrid_filtered)
    return result.sort_values("hybrid_score", ascending=False).reset_index(drop=True)


def recommend_tfidf(course_id: int, top_n: int, state: dict) -> pd.DataFrame:
    """Model 5: TF-IDF Content-Based."""
    dp = state["df_processed"]
    row = dp[dp["course_id"] == course_id]
    if row.empty:
        return pd.DataFrame()
    course_name = row["course_name"].iloc[0]
    ci = state["tfidf_course_index"]
    if course_name not in ci.index:
        return pd.DataFrame()
    idx = ci[course_name]
    cosine_sim = linear_kernel(state["tfidf_matrix"][idx], state["tfidf_matrix"]).flatten()
    sim_indices = cosine_sim.argsort()[::-1][1: top_n + 1]
    rec_names = state["course_df"].iloc[sim_indices]["course_name"].values
    rec_sims = cosine_sim[sim_indices]
    result = state["course_catalog"][
        state["course_catalog"]["course_id"].isin(
            state["course_df"].iloc[sim_indices]["course_id"].values
        )
    ].copy()
    name_to_sim = dict(zip(rec_names, rec_sims))
    result["tfidf_similarity"] = result["course_id"].apply(
        lambda cid: float(
            name_to_sim.get(
                state["df_processed"][state["df_processed"]["course_id"] == cid]["course_name"].iloc[0]
                if not state["df_processed"][state["df_processed"]["course_id"] == cid].empty
                else "", 0
            )
        )
    )
    return result.sort_values("tfidf_similarity", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📚 Course Recommender")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 EDA Dashboard", "🎯 Course-to-Course", "👤 User Recommendations"],
    )
    st.markdown("---")
    st.caption("5-Model Recommendation Engine")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA UPLOAD (shared across pages)
# ═══════════════════════════════════════════════════════════════════════════════
uploaded = st.sidebar.file_uploader(
    "Upload Dataset (CSV)", type=["csv"], key="main_upload"
)

state = None
if uploaded:
    with st.spinner("⚙️ Preprocessing data and building models… (first load may take ~30s)"):
        state = load_and_preprocess(uploaded)
    st.sidebar.success(f"✅ {len(state['df']):,} rows loaded")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📚 Online Course Recommendation System")
    st.markdown(
        """
        An intelligent **multi-model recommendation engine** that suggests courses to learners based on
        their interests, past enrollments, and engagement levels.

        ---
        ### 🎯 Business Objective
        Reduce choice overload on e-learning platforms by delivering personalised course suggestions,
        improving learner satisfaction and completion rates.

        ### 🤖 5 Models Available

        | # | Model | Type |
        |---|---|---|
        | 1 | KNN on PCA Features | Content-Based |
        | 2 | User-Based KNN | Collaborative Filtering |
        | 3 | SVD Matrix Factorization | Collaborative Filtering |
        | 4 | Hybrid (CF + CBF, α=0.7) | Hybrid |
        | 5 | TF-IDF + Cosine Similarity | Content-Based (NLP) |

        ### 📂 How to Use
        1. **Upload** the CSV file using the sidebar.
        2. Go to **Course-to-Course** to find similar courses (Models 1, 5).
        3. Go to **User Recommendations** for personalised picks (Models 2, 3, 4).
        4. Explore **EDA Dashboard** for data insights.
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", "1,00,000")
    col2.metric("Features", "14")
    col3.metric("Engineered Features", "7")
    col4.metric("Recommendation Models", "5")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.title("📊 EDA Dashboard")
    if state is None:
        st.info("👆 Upload the dataset from the sidebar to begin.")
        st.stop()

    df = state["df"]
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribution", "Correlation", "Categorical", "PCA Variance"]
    )

    with tab1:
        col_sel = st.selectbox("Select Numerical Feature", NUMERICAL_FEATURES)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].hist(df[col_sel], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].axvline(df[col_sel].mean(), color="red", linestyle="--", label="Mean")
        axes[0].axvline(df[col_sel].median(), color="green", linestyle="--", label="Median")
        axes[0].set_title(f"Distribution of {col_sel}")
        axes[0].legend()
        axes[1].boxplot(df[col_sel], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="lightblue"))
        axes[1].set_title(f"Boxplot of {col_sel}")
        st.pyplot(fig)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{df[col_sel].mean():.2f}")
        c2.metric("Skewness", f"{df[col_sel].skew():.3f}")
        c3.metric("Kurtosis", f"{df[col_sel].kurtosis():.3f}")

    with tab2:
        st.subheader("Correlation Heatmap — Numerical Features")
        corr = df[NUMERICAL_FEATURES].corr()
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax2, square=True)
        st.pyplot(fig2)

    with tab3:
        cat_sel = st.selectbox(
            "Select Categorical Feature",
            ["difficulty_level", "certification_offered", "study_material_available"],
        )
        vc = df[cat_sel].value_counts()
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        axes3[0].bar(vc.index, vc.values, color="coral", edgecolor="black", alpha=0.8)
        axes3[0].set_title(f"Count — {cat_sel}")
        axes3[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                     colors=sns.color_palette("pastel"), startangle=90)
        axes3[1].set_title(f"Proportion — {cat_sel}")
        st.pyplot(fig3)

    with tab4:
        st.subheader("PCA — Explained Variance")
        pca_full = state["pca_full"]
        ev = pca_full.explained_variance_ratio_
        cv = np.cumsum(ev)
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
        axes4[0].plot(range(1, len(ev) + 1), ev, "bo-")
        axes4[0].set_title("Scree Plot")
        axes4[0].set_xlabel("Component")
        axes4[0].set_ylabel("Explained Variance Ratio")
        axes4[1].plot(range(1, len(cv) + 1), cv, "ro-")
        axes4[1].axhline(0.95, color="green", linestyle="--", label="95% threshold")
        axes4[1].set_title("Cumulative Explained Variance")
        axes4[1].set_xlabel("Component")
        axes4[1].legend()
        st.pyplot(fig4)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: COURSE-TO-COURSE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Course-to-Course":
    st.title("🎯 Course-to-Course Recommendations")
    if state is None:
        st.info("👆 Upload the dataset from the sidebar to begin.")
        st.stop()

    catalog = state["course_catalog"]
    model_choice = st.selectbox(
        "Select Model",
        ["Model 1 — Content-Based KNN (PCA + Cosine)",
         "Model 5 — TF-IDF Content-Based (NLP)"],
    )
    top_n = st.slider("Number of Recommendations", 3, 15, 5)

    course_ids = sorted(catalog["course_id"].unique())
    course_id = st.selectbox("Select Course ID", course_ids)

    # Show selected course info
    sel = catalog[catalog["course_id"] == course_id].iloc[0]
    st.markdown(f"**Selected:** `{sel['course_name']}` | "
                f"Difficulty: `{sel['difficulty_level']}` | "
                f"Rating: `{sel['rating']:.2f}` | "
                f"Price: `${sel['course_price']:.0f}`")

    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Finding similar courses…"):
            if "KNN" in model_choice:
                result = recommend_knn(course_id, top_n, state)
                score_col = "cosine_similarity"
            else:
                result = recommend_tfidf(course_id, top_n, state)
                score_col = "tfidf_similarity"

        if result.empty:
            st.warning("No recommendations found for this course.")
        else:
            st.success(f"Top {len(result)} recommendations:")
            display_cols = ["course_id", "course_name", "difficulty_level",
                            "certification_offered", "rating", "course_price", score_col]
            display_cols = [c for c in display_cols if c in result.columns]
            st.dataframe(result[display_cols].style.format(
                {score_col: "{:.4f}", "rating": "{:.2f}", "course_price": "${:.0f}"}
            ), use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(result["course_name"].str[:40], result[score_col], color="steelblue")
            ax.set_xlabel(score_col.replace("_", " ").title())
            ax.set_title("Recommendation Scores")
            ax.invert_yaxis()
            st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: USER RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤 User Recommendations":
    st.title("👤 Personalised User Recommendations")
    if state is None:
        st.info("👆 Upload the dataset from the sidebar to begin.")
        st.stop()

    uim = state["user_item_matrix"]
    model_choice = st.selectbox(
        "Select Model",
        ["Model 2 — User-Based Collaborative Filtering (KNN)",
         "Model 3 — SVD Matrix Factorization",
         "Model 4 — Hybrid (CF + CBF)"],
    )
    top_n = st.slider("Number of Recommendations", 3, 15, 5)

    user_ids = sorted(uim.index.tolist())
    user_id = st.selectbox("Select User ID", user_ids)

    user_history = uim.loc[user_id]
    taken = user_history[user_history > 0]
    st.markdown(f"**User {user_id}** has rated **{len(taken)}** courses.")

    if st.button("🎯 Get Personalised Recommendations", type="primary"):
        with st.spinner("Generating recommendations…"):
            if "User-Based" in model_choice:
                result = recommend_user_based(user_id, top_n, state)
                score_col = "predicted_score"
            elif "SVD" in model_choice:
                result = recommend_svd(user_id, top_n, state)
                score_col = "predicted_rating"
            else:
                alpha_val = st.sidebar.slider("Hybrid Alpha (CF weight)", 0.0, 1.0, ALPHA)
                result = recommend_hybrid(user_id, top_n, ALPHA, state)
                score_col = "hybrid_score"

        if result.empty:
            st.warning("No recommendations found for this user.")
        else:
            st.success(f"Top {len(result)} personalised recommendations:")
            display_cols = ["course_id", "course_name", "difficulty_level",
                            "certification_offered", "rating", "course_price", score_col]
            display_cols = [c for c in display_cols if c in result.columns]
            st.dataframe(result[display_cols].style.format(
                {score_col: "{:.4f}", "rating": "{:.2f}", "course_price": "${:.0f}"}
            ), use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(result["course_name"].str[:40], result[score_col], color="coral")
            ax.set_xlabel(score_col.replace("_", " ").title())
            ax.set_title("Personalised Recommendation Scores")
            ax.invert_yaxis()
            st.pyplot(fig)

    # ── History ───────────────────────────────────────────────────────────────
    with st.expander("📋 View User's Rating History"):
        rated_courses = taken.reset_index()
        rated_courses.columns = ["course_id", "rating"]
        merged = rated_courses.merge(state["course_catalog"], on="course_id", how="left")
        st.dataframe(merged[["course_id", "course_name", "difficulty_level", "rating_x"]].rename(
            columns={"rating_x": "user_rating"}
        ), use_container_width=True)
