import pandas as pd
import numpy as np
import random

# Number of accounts
n = 500
data = []

for i in range(n):
    # Randomly generate base features
    followers = np.random.randint(0, 15000)
    following = np.random.randint(0, 15000)
    posts = np.random.randint(0, 1000)
    profile_pic = np.random.choice([0, 1])  # 0 = No, 1 = Yes
    has_posts = 1 if posts > 0 else 0

    # Derived features
    follower_following_ratio = followers / (following + 1)
    posts_per_follower = posts / (followers + 1)

    # Default label and confidence
    label = 0
    confidence = 0.5

    # ✅ 1. Balanced followers & following (~100), no posts/pic → 70% real
    if 50 <= followers <= 150 and 50 <= following <= 150 and profile_pic == 0 and has_posts == 0:
        label = 1
        confidence = np.random.uniform(0.65, 0.75)

    # ✅ 2. Balanced followers & following (~100), with posts + pic → 80% real
    elif 50 <= followers <= 150 and 50 <= following <= 150 and profile_pic == 1 and has_posts == 1:
        label = 1
        confidence = np.random.uniform(0.78, 0.85)

    # ✅ 3. Low followers, very high following, no posts/pic → Fake 90–100%
    elif followers < 200 and following > 1000 and profile_pic == 0 and has_posts == 0:
        label = 0
        confidence = np.random.uniform(0.9, 1.0)

    # ✅ 4. High followers, low following, has posts & profile pic → 90–100% real
    elif followers > 1000 and following < 500 and profile_pic == 1 and has_posts == 1:
        label = 1
        confidence = np.random.uniform(0.9, 1.0)

    # ✅ 5. Low followers, very high following → Fake (default low confidence)
    elif followers < 200 and following > 1000:
        label = 0
        confidence = np.random.uniform(0.0, 0.4)

    # ✅ 6. Low followers, high following, has posts >10 + profile pic → Real 80%
    elif followers < 500 and following > 1000 and posts > 10 and profile_pic == 1:
        label = 1
        confidence = np.random.uniform(0.78, 0.85)

    # ✅ 7. Low followers, high following, has posts <10 + profile pic → Real 60%
    elif followers < 500 and following > 1000 and posts < 10 and profile_pic == 1:
        label = 1
        confidence = np.random.uniform(0.58, 0.65)

    # ✅ 8. Catch-all random accounts → 50–70%
    else:
        label = np.random.choice([0, 1])
        confidence = np.random.uniform(0.5, 0.7)

    data.append({
        "username": f"user_{i+1}",
        "followers": followers,
        "following": following,
        "posts": posts,
        "follower_following_ratio": follower_following_ratio,
        "posts_per_follower": posts_per_follower,
        "profile_pic": profile_pic,
        "has_posts": has_posts,
        "confidence": confidence,
        "label": label
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv("realistic_dataset.csv", index=False)
print("✅ realistic_dataset.csv generated successfully with 8 realistic behavior-based conditions!")
