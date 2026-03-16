import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Paths and constants
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
DATASET_PATH = "realistic_dataset.csv"

REQUIRED_BASE_COLS = ["followers", "following", "posts", "profile_pic", "label"]
FEATURE_COLS = [
    "followers",
    "following",
    "posts",
    "profile_pic",
    "follower_following_ratio",
    "posts_per_follower",
    "has_posts",
]

# -------------------------------
st.set_page_config(page_title="Social Media Account Detector", page_icon="🔑", layout="wide")
st.markdown(
    "<h1 style='text-align:center; color:white; padding:10px;'>🔑 Social Media Account Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Session defaults
defaults = {
    "df": None,
    "model": None,
    "scaler": None,
    "followers": 0,
    "following": 0,
    "posts": 0,
    "profile_pic": "No",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def generate_default_dataset(n=500):
    data = []
    for i in range(n):
        choice = np.random.choice(
            [
                "balanced",
                "high_foll_low_following",
                "low_foll_high_following_no_posts",
                "low_foll_high_following_with_posts",
            ]
        )

        if choice == "balanced":
            followers = np.random.randint(90, 110)
            following = np.random.randint(90, 110)
            posts = 0
            profile_pic = 0
            label = 1
            confidence = np.random.uniform(0.65, 0.75)

        elif choice == "high_foll_low_following":
            followers = np.random.randint(5000, 10000)
            following = np.random.randint(50, 500)
            posts = np.random.randint(50, 500)
            profile_pic = 1
            label = 1
            confidence = np.random.uniform(0.90, 1.0)

        elif choice == "low_foll_high_following_no_posts":
            followers = np.random.randint(0, 200)
            following = np.random.randint(5000, 10000)
            posts = 0
            profile_pic = 0
            label = 0
            confidence = np.random.uniform(0.90, 1.0)

        else:
            followers = np.random.randint(0, 200)
            following = np.random.randint(5000, 10000)
            posts = np.random.randint(10, 100)
            profile_pic = 1
            label = 1
            confidence = np.random.uniform(0.78, 0.85)

        has_posts = 1 if posts > 0 else 0
        data.append(
            {
                "username": f"user_{i+1}",
                "followers": followers,
                "following": following,
                "posts": posts,
                "profile_pic": profile_pic,
                "has_posts": has_posts,
                "follower_following_ratio": followers / (following + 1),
                "posts_per_follower": posts / (followers + 1),
                "label": label,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(data)


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize headers
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Common alias mapping
    alias_map = {
        "follower": "followers",
        "follower_count": "followers",
        "following_count": "following",
        "post": "posts",
        "post_count": "posts",
        "profilepic": "profile_pic",
        "is_real": "label",
        "class": "label",
        "target": "label",
    }
    df = df.rename(columns={k: v for k, v in alias_map.items() if k in df.columns})

    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found: {list(df.columns)}")

    # Numeric columns
    for col in ["followers", "following", "posts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # profile_pic -> 0/1
    if df["profile_pic"].dtype == object:
        mapped_pp = (
            df["profile_pic"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        )
        numeric_pp = pd.to_numeric(df["profile_pic"], errors="coerce")
        df["profile_pic"] = mapped_pp.fillna(numeric_pp).fillna(0).astype(int)
    else:
        df["profile_pic"] = pd.to_numeric(df["profile_pic"], errors="coerce").fillna(0).astype(int)

    # label -> 0/1
    if df["label"].dtype == object:
        mapped_label = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"real": 1, "fake": 0, "yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
        )
        numeric_label = pd.to_numeric(df["label"], errors="coerce")
        df["label"] = mapped_label.fillna(numeric_label)
    else:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df["label"] = df["label"].fillna(0).astype(int)
    df["label"] = df["label"].apply(lambda x: 1 if x == 1 else 0)

    # Derived features
    if "has_posts" not in df.columns:
        df["has_posts"] = (df["posts"] > 0).astype(int)
    else:
        df["has_posts"] = pd.to_numeric(df["has_posts"], errors="coerce").fillna(0).astype(int)

    if "follower_following_ratio" not in df.columns:
        df["follower_following_ratio"] = df["followers"] / (df["following"] + 1)
    else:
        df["follower_following_ratio"] = pd.to_numeric(df["follower_following_ratio"], errors="coerce").fillna(
            df["followers"] / (df["following"] + 1)
        )

    if "posts_per_follower" not in df.columns:
        df["posts_per_follower"] = df["posts"] / (df["followers"] + 1)
    else:
        df["posts_per_follower"] = pd.to_numeric(df["posts_per_follower"], errors="coerce").fillna(
            df["posts"] / (df["followers"] + 1)
        )

    if "username" not in df.columns:
        df["username"] = [f"user_{i+1}" for i in range(len(df))]

    return df


def ensure_dataset():
    if st.session_state.df is not None:
        return st.session_state.df

    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            df = normalize_dataset(df)
            st.session_state.df = df
            return df
        except Exception as e:
            st.warning(f"Existing dataset is invalid ({e}). Generating default dataset...")

    df = generate_default_dataset()
    df.to_csv(DATASET_PATH, index=False)
    st.session_state.df = df
    st.success("✅ Default dataset generated successfully!")
    return df


def load_model_scaler():
    if st.session_state.model is not None and st.session_state.scaler is not None:
        return st.session_state.model, st.session_state.scaler

    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        st.error("❌ Model files not found. Please train the model first.")
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        st.session_state.model = model
        st.session_state.scaler = scaler
        return model, scaler
    except Exception as e:
        st.error(f"❌ Failed to load model/scaler: {e}")
        st.info("Delete old model files and retrain in 'Train Model'.")
        return None, None


def predict_probs_with_class_map(model, X_scaled):
    # Ensures probabilities map correctly to class 0=fake, 1=real
    probs = model.predict_proba(X_scaled)[0]
    class_to_prob = {int(c): float(p) for c, p in zip(model.classes_, probs)}
    fake_prob = class_to_prob.get(0, 0.0)
    real_prob = class_to_prob.get(1, 0.0)
    total = fake_prob + real_prob
    if total == 0:
        return 0.5, 0.5
    return fake_prob / total, real_prob / total


# -------------------------------
# Sidebar
st.sidebar.title("📌 Sections")
section = st.sidebar.radio(
    "Select a Section",
    ["Dataset Upload / Generate", "Train Model", "Check Single Account", "Dataset Analysis", "Dashboard"],
)

# -------------------------------
# 1) Dataset Upload / Generate
if section == "Dataset Upload / Generate":
    st.header("📥 Upload or Generate Dataset")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = normalize_dataset(df)
            st.session_state.df = df
            df.to_csv(DATASET_PATH, index=False)
            st.success("✅ Dataset loaded and normalized successfully!")
        except Exception as e:
            st.error(f"❌ Invalid dataset: {e}")
            st.stop()
    else:
        df = ensure_dataset()

    st.dataframe(df.head(10))

# -------------------------------
# 2) Train Model
elif section == "Train Model":
    st.header("🧠 Train Model")
    df = ensure_dataset()

    if st.button("Train Model"):
        X = df[FEATURE_COLS]
        y = df["label"].astype(int)

        if y.nunique() < 2:
            st.error("❌ Training needs both classes (real and fake). Please upload a balanced dataset.")
            st.stop()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        st.session_state.model = model
        st.session_state.scaler = scaler

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"🎉 Model trained successfully! Accuracy: {acc * 100:.2f}%")
        st.text(classification_report(y_test, y_pred))

# -------------------------------
# 3) Check Single Account
elif section == "Check Single Account":
    st.header("🔍 Check Single Account")

    st.session_state.followers = st.number_input(
        "Followers", min_value=0, max_value=10_000_000_000, value=int(st.session_state.followers)
    )
    st.session_state.following = st.number_input(
        "Following", min_value=0, max_value=10_000_000_000, value=int(st.session_state.following)
    )
    st.session_state.posts = st.number_input(
        "Posts", min_value=0, max_value=100_000, value=int(st.session_state.posts)
    )
    st.session_state.profile_pic = st.selectbox(
        "Profile Picture? 🤳",
        ["No", "Yes"],
        index=0 if st.session_state.profile_pic == "No" else 1,
    )

    has_posts_input = 1 if st.session_state.posts > 0 else 0

    if st.button("Predict Account"):
        model, scaler = load_model_scaler()
        if model is None or scaler is None:
            st.stop()

        input_data = pd.DataFrame(
            [
                [
                    st.session_state.followers,
                    st.session_state.following,
                    st.session_state.posts,
                    1 if st.session_state.profile_pic == "Yes" else 0,
                    st.session_state.followers / (st.session_state.following + 1),
                    st.session_state.posts / (st.session_state.followers + 1),
                    has_posts_input,
                ]
            ],
            columns=FEATURE_COLS,
        )

        input_scaled = scaler.transform(input_data)
        fake_prob, real_prob = predict_probs_with_class_map(model, input_scaled)

        # Custom logic overrides
        if 50 <= st.session_state.followers <= 150 and 50 <= st.session_state.following <= 150:
            if st.session_state.profile_pic == "No" and st.session_state.posts == 0:
                fake_prob, real_prob = 0.30, 0.70
            elif st.session_state.profile_pic == "Yes" and st.session_state.posts > 0:
                fake_prob, real_prob = 0.20, 0.80
        elif (
            st.session_state.followers < 500
            and st.session_state.following > 1000
            and st.session_state.posts < 10
            and st.session_state.profile_pic == "Yes"
        ):
            fake_prob, real_prob = 0.40, 0.60
        elif (
            st.session_state.followers < 500
            and st.session_state.following > 1000
            and st.session_state.posts > 10
            and st.session_state.profile_pic == "Yes"
        ):
            fake_prob, real_prob = 0.20, 0.80
        elif (
            st.session_state.followers < 200
            and st.session_state.following > 1000
            and st.session_state.posts == 0
            and st.session_state.profile_pic == "No"
        ):
            fake_prob, real_prob = 0.95, 0.05

        is_real = real_prob > fake_prob
        st.markdown(f"### Result: {'🟢 REAL' if is_real else '🔴 FAKE'}")

        st.markdown("Confidence Levels:")
        st.progress(float(real_prob))
        st.markdown(f"🟢 Real: {real_prob * 100:.2f}%")
        st.progress(float(fake_prob))
        st.markdown(f"🔴 Fake: {fake_prob * 100:.2f}%")

        pie_df = pd.DataFrame(
            {"Type": ["Real", "Fake"], "Probability": [real_prob * 100, fake_prob * 100]}
        )
        fig_pie = px.pie(
            pie_df,
            names="Type",
            values="Probability",
            color="Type",
            color_discrete_map={"Real": "green", "Fake": "red"},
            title="Prediction Probability",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# 4) Dataset Analysis
elif section == "Dataset Analysis":
    st.header("📊 Dataset Analysis")
    df = ensure_dataset()

    if st.checkbox("Show Followers Distribution 📈"):
        fig_follow = px.histogram(
            df,
            x="followers",
            nbins=30,
            color="label",
            color_discrete_map={0: "red", 1: "green"},
            title="📊 Followers Distribution by Account Type",
        )
        st.plotly_chart(fig_follow, use_container_width=True)

    if st.checkbox("Show Follower/Following Ratio vs Posts per Follower 📊"):
        fig_ratio = px.scatter(
            df,
            x="follower_following_ratio",
            y="posts_per_follower",
            color="label",
            color_discrete_map={0: "red", 1: "green"},
            title="📈 Follower/Following Ratio vs Posts per Follower",
            size="posts",
            hover_data=["username"],
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

# -------------------------------
# 5) Dashboard
elif section == "Dashboard":
    st.header("📊 Dataset Dashboard")
    df = ensure_dataset()

    model, scaler = load_model_scaler()
    if model is None or scaler is None:
        st.warning("⚠ Model not trained. Please train in section 2 first.")
        st.stop()

    total_accounts = len(df)
    total_real = int(df["label"].sum())
    total_fake = total_accounts - total_real

    st.markdown(f"### Total Accounts: {total_accounts} ✅")
    st.markdown(f"🟢 Real Accounts: {total_real} ({(total_real / total_accounts) * 100:.2f}%)")
    st.markdown(f"🔴 Fake Accounts: {total_fake} ({(total_fake / total_accounts) * 100:.2f}%)")

    pie_df = pd.DataFrame({"Type": ["Real", "Fake"], "Count": [total_real, total_fake]})
    fig_pie = px.pie(
        pie_df,
        names="Type",
        values="Count",
        color="Type",
        color_discrete_map={"Real": "green", "Fake": "red"},
        title="Real vs Fake Accounts Overview",
        hole=0.3,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    X_dash = df[FEATURE_COLS]
    X_dash_scaled = scaler.transform(X_dash)
    proba = model.predict_proba(X_dash_scaled)

    classes = list(model.classes_)
    if 0 in classes:
        idx_fake = classes.index(0)
        fake_conf = proba[:, idx_fake]
    else:
        fake_conf = np.zeros(len(df))

    df_dash = df.copy()
    df_dash["fake_confidence"] = fake_conf

    top_fake = df_dash.sort_values("fake_confidence", ascending=False).head(5)
    st.markdown("### 🔝 Top 5 Most Likely Fake Accounts")
    st.dataframe(
        top_fake[
            ["username", "followers", "following", "posts", "profile_pic", "has_posts", "fake_confidence"]
        ]
    )

    fig_bar = px.bar(
        top_fake,
        x="username",
        y="fake_confidence",
        color="fake_confidence",
        color_continuous_scale="Reds",
        title="Top 5 Fake Account Confidence",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
