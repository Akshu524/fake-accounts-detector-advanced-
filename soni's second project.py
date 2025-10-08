def is_fake_account(username, followers, following, posts, has_profile_pic):
    # Rule 1: very few followers but follows many
    if followers < 10 and following > 500:
        return True
    
    # Rule 2: almost no posts
    if posts < 2:
        return True
    
    # Rule 3: no profile picture
    if has_profile_pic == 0:
        return True
    
    # Rule 4: suspicious username (too many digits)
    digits = sum(c.isdigit() for c in username)
    if digits > len(username) // 2:
        return True
    
    # Rule 5: suspicious keywords in username
    suspicious_words = ["free", "win", "money", "follow"]
    for word in suspicious_words:
        if word in username.lower():
            return True
    
    # Otherwise assume real
    return False


print("🔍 Fake Account Detector")
print("Type 'exit' as username to quit.\n")

while True:
    username = input("Enter username: ")
    if username.lower() == "exit":
        print("Exiting... 👋")
        break

    followers = int(input("Enter number of followers: "))
    following = int(input("Enter number of following: "))
    posts = int(input("Enter number of posts: "))
    has_profile_pic = int(input("Has profile pic? (1=yes, 0=no): "))

    if is_fake_account(username, followers, following, posts, has_profile_pic):
        print(f"🚨 {username} is a FAKE account!\n")
    else:
        print(f"✅ {username} looks REAL.\n")