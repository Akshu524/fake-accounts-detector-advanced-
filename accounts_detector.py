import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import os

# -------------------------------
st.set_page_config(page_title="Social Media Account Detector", page_icon="🔑", layout="wide")
st.markdown("<h1 style='text-align:center; color:white; padding:10px;'>🔑 Social Media Account Detector</h1>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
for key, default in {
    "df": None, "model": None, "scaler": None,
    "followers": 0, "following": 0, "posts": 0, "profile_pic": "No"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------------
def ensure_dataset():
    """Ensures dataset exists or generates a new one."""
    if st.session_state.df is not None:
        return st.session_state.df

    if os.path.exists("realistic_dataset.csv"):
        df = pd.read_csv("data/realistic_data.csv")
        st.session_state.df = df
        return df

    st.info("No dataset found. Generating new one automatically…")
    n = 500
    data = []
    for i in range(n):
        choice = np.random.choice([
            "balanced",
            "high_foll_low_following",
            "low_foll_high_following_no_posts",
            "low_foll_high_following_with_posts"
        ])
        if choice == "balanced":
            followers = np.random.randint(90, 110)
            following = np.random.randint(90, 110)
            posts = 0
            profile_pic = 0
            label = 1
        elif choice == "high_foll_low_following":
            followers = np.random.randint(5000, 10000)
            following = np.random.randint(50, 500)
            posts = np.random.randint(50, 500)
            profile_pic = 1
            label = 1
        elif choice == "low_foll_high_following_no_posts":
            followers = np.random.randint(0, 200)
            following = np.random.randint(5000, 10000)
            posts = 0
            profile_pic = 0
            label = 0
        else:
            followers = np.random.randint(0, 200)
            following = np.random.randint(5000, 10000)
            posts = np.random.randint(10, 100)
            profile_pic = 1
            label = 1

        has_posts = 1 if posts > 0 else 0
        confidence = 0.5
        if choice == "balanced":
            confidence = np.random.uniform(0.65, 0.75)
        elif choice == "high_foll_low_following":
            confidence = np.random.uniform(0.90, 1.0)
        elif choice == "low_foll_high_following_no_posts":
            confidence = np.random.uniform(0.90, 1.0)
        else:
            confidence = np.random.uniform(0.78, 0.85)

        data.append({
            "username": f"user_{i+1}",
            "followers": followers,
            "following": following,
            "posts": posts,
            "profile_pic": profile_pic,
            "has_posts": has_posts,
            "follower_following_ratio": followers/(following+1),
            "posts_per_follower": posts/(followers+1),
            "label": label,
            "confidence": confidence
        })

    df = pd.DataFrame(data)
    df.to_csv("realistic_dataset.csv", index=False)
    st.session_state.df = df
    st.success("✅ Default dataset generated successfully!")
    return df

# -------------------------------
# Sidebar Navigation
st.sidebar.title("📌 Sections")
section = st.sidebar.radio(
    "Select a Section",
    ["Dataset Upload / Generate", "Train Model", "Check Single Account", "Dataset Analysis", "Dashboard"]
)

# -------------------------------
# 1️⃣ Dataset Upload / Generate
if section == "Dataset Upload / Generate":
    st.header("📥 Upload or Generate Dataset")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
    else:
        df = ensure_dataset()
    st.session_state.df = df
    st.dataframe(df.head(10))

# -------------------------------
# 2️⃣ Train Model
elif section == "Train Model":
    st.header("🧠 Train Model")
    df = ensure_dataset()

    if st.button("Train Model"):
        X = df[["followers", "following", "posts", "profile_pic", "follower_following_ratio", "posts_per_follower", "has_posts"]]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)

        joblib.dump(model, "model.joblib")
        joblib.dump(scaler, "scaler.joblib")

        st.session_state.model = model
        st.session_state.scaler = scaler

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"🎉 Model trained successfully! Accuracy: {acc*100:.2f}%")
        st.text(classification_report(y_test, y_pred))

# -------------------------------
# 3️⃣ Check Single Account
elif section == "Check Single Account":
    st.header("🔍 Check Single Account")

    # Keep last inputs in session state
    st.session_state.followers = st.number_input("Followers", 0, 10000000000, value=st.session_state.followers)
    st.session_state.following = st.number_input("Following", 0, 10000000000, value=st.session_state.following)
    st.session_state.posts = st.number_input("Posts", 0, 100000, value=st.session_state.posts)
    st.session_state.profile_pic = st.selectbox("Profile Picture? 🤳", ["No", "Yes"], index=0 if st.session_state.profile_pic == "No" else 1)
    has_posts_input = 1 if st.session_state.posts > 0 else 0

    if st.button("Predict Account"):
        try:
            model = st.session_state.model or joblib.load("models/model.joblib")
            scaler = st.session_state.scaler or joblib.load("scaler.joblib")
        except FileNotFoundError:
            st.error("❌ Please train the model first in section 2.")
            st.stop()

        input_data = pd.DataFrame([[st.session_state.followers, st.session_state.following, st.session_state.posts,
                                   1 if st.session_state.profile_pic == "Yes" else 0,
                                   st.session_state.followers / (st.session_state.following + 1),
                                   st.session_state.posts / (st.session_state.followers + 1),
                                   has_posts_input]],
                                   columns=["followers", "following", "posts", "profile_pic",
                                            "follower_following_ratio", "posts_per_follower", "has_posts"])
        input_scaled = scaler.transform(input_data)
        probability = model.predict_proba(input_scaled)[0]

        # Custom logic
        if 50 <= st.session_state.followers <= 150 and 50 <= st.session_state.following <= 150:
            if st.session_state.profile_pic == "No" and st.session_state.posts == 0:
                probability[1] = 0.70; probability[0] = 0.30
            elif st.session_state.profile_pic == "Yes" and st.session_state.posts > 0:
                probability[1] = 0.80; probability[0] = 0.20
        elif st.session_state.followers < 500 and st.session_state.following > 1000 and st.session_state.posts < 10 and st.session_state.profile_pic == "Yes":
            probability[1] = 0.60; probability[0] = 0.40
        elif st.session_state.followers < 500 and st.session_state.following > 1000 and st.session_state.posts > 10 and st.session_state.profile_pic == "Yes":
            probability[1] = 0.80; probability[0] = 0.20
        elif st.session_state.followers < 200 and st.session_state.following > 1000 and st.session_state.posts == 0 and st.session_state.profile_pic == "No":
            probability[1] = 0.05; probability[0] = 0.95

        is_real = probability[1] > probability[0]
        st.markdown(f"### Result: {'🟢 REAL' if is_real else '🔴 FAKE'}")

        st.markdown("Confidence Levels:")
        st.progress(probability[1])
        st.markdown(f"🟢 Real: {probability[1]*100:.2f}%")
        st.progress(probability[0])
        st.markdown(f"🔴 Fake: {probability[0]*100:.2f}%")

        pie_df = pd.DataFrame({'Type': ['Real', 'Fake'], 'Probability': [probability[1]*100, probability[0]*100]})
        fig_pie = px.pie(pie_df, names='Type', values='Probability',
                         color='Type', color_discrete_map={'Real': 'green', 'Fake': 'red'},
                         title="Prediction Probability")
        st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# 4️⃣ Dataset Analysis
elif section == "Dataset Analysis":
    st.header("📊 Dataset Analysis")
    df = ensure_dataset()

    if st.checkbox("Show Followers Distribution 📈"):
        fig_follow = px.histogram(df, x="followers", nbins=30, color="label",
                                  color_discrete_map={0: "red", 1: "green"},
                                  title="📊 Followers Distribution by Account Type")
        st.plotly_chart(fig_follow, use_container_width=True)

    if st.checkbox("Show Follower/Following Ratio vs Posts per Follower 📊"):
        fig_ratio = px.scatter(df, x="follower_following_ratio", y="posts_per_follower",
                               color="label", color_discrete_map={0: "red", 1: "green"},
                               title="📈 Follower/Following Ratio vs Posts per Follower",
                               size="posts", hover_data=["username"])
        st.plotly_chart(fig_ratio, use_container_width=True)

# -------------------------------
# 5️⃣ Dashboard
elif section == "Dashboard":
    st.header("📊 Dataset Dashboard")
    df = ensure_dataset()

    try:
        model = st.session_state.model or joblib.load("model.joblib")
        scaler = st.session_state.scaler or joblib.load("scaler.joblib")
    except:
        st.warning("⚠ Model not trained. Please train in section 2 first.")
        st.stop()

    total_accounts = len(df)
    total_real = df['label'].sum()
    total_fake = total_accounts - total_real
    st.markdown(f"### Total Accounts: {total_accounts} ✅")
    st.markdown(f"🟢 Real Accounts: {total_real} ({(total_real/total_accounts)*100:.2f}%)")
    st.markdown(f"🔴 Fake Accounts: {total_fake} ({(total_fake/total_accounts)*100:.2f}%)")

    pie_df = pd.DataFrame({'Type': ['Real', 'Fake'], 'Count': [total_real, total_fake]})
    fig_pie = px.pie(pie_df, names='Type', values='Count',
                     color='Type', color_discrete_map={'Real': 'green', 'Fake': 'red'},
                     title="Real vs Fake Accounts Overview", hole=0.3)
    st.plotly_chart(fig_pie, use_container_width=True)

    X_dash = df[["followers", "following", "posts", "profile_pic", "follower_following_ratio", "posts_per_follower", "has_posts"]]
    X_dash_scaled = scaler.transform(X_dash)
    proba = model.predict_proba(X_dash_scaled)
    fake_conf = [p[0] for p in proba]
    df['fake_confidence'] = fake_conf

    top_fake = df.sort_values('fake_confidence', ascending=False).head(5)
    st.markdown("### 🔝 Top 5 Most Likely Fake Accounts")
    st.dataframe(top_fake[['username', 'followers', 'following', 'posts', 'profile_pic', 'has_posts', 'fake_confidence']])

    fig_bar = px.bar(top_fake, x='username', y='fake_confidence',
                     color='fake_confidence', color_continuous_scale='Reds',
                     title="Top 5 Fake Account Confidence")
    st.plotly_chart(fig_bar, use_container_width=True)
