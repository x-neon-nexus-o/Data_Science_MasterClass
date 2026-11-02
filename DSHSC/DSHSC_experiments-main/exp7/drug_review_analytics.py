import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------------------
# Step 1: Load Healthcare Dataset (CO1: Healthcare Data)
# -------------------------------------------------------------
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drug_reviews.csv")
df = pd.read_csv(dataset_path)

print("=" * 70)
print("EXPERIMENT 7: DRUG REVIEW ANALYTICS & SENTIMENT MINING")
print("Course Outcome: CO1 - Clean, Integrate, and Transform Healthcare Data")
print("=" * 70)
print(f"Total Patient Drug Reviews: {len(df)}")
print("\nSample Raw Healthcare Data:")
print(df[["Drug_Name", "Condition", "Rating", "Review"]].head(3))

# -------------------------------------------------------------
# Step 2: Clean Healthcare Text Data (CO1: Data Cleaning)
# -------------------------------------------------------------
def clean_review_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation, numbers, and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Remove redundant whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Cleaned_Review"] = df["Review"].apply(clean_review_text)

print("\n--- Step 2: Cleaned Text Sample ---")
print("Raw     :", df["Review"].iloc[1])
print("Cleaned :", df["Cleaned_Review"].iloc[1])

# -------------------------------------------------------------
# Step 3: Transform Healthcare Data (CO1: Data Transformation)
# -------------------------------------------------------------
# Transform ratings into Sentiment Classes:
# Rating >= 7 -> Positive, Rating <= 4 -> Negative (Filter neutral 5-6)
df = df[df["Rating"] != 6].copy()
df["Sentiment"] = df["Rating"].apply(lambda r: "Positive" if r >= 7 else "Negative")

print("\n--- Step 3: Transformed Sentiment Distribution ---")
print(df["Sentiment"].value_counts())

# -------------------------------------------------------------
# Step 4: Integrate Healthcare Insights (CO1: Integration)
# -------------------------------------------------------------
drug_summary = df.groupby("Drug_Name").agg(
    Avg_Rating=("Rating", "mean"),
    Total_Reviews=("Rating", "count"),
    Positive_Reviews=("Sentiment", lambda s: (s == "Positive").sum())
).reset_index()

print("\n--- Step 4: Integrated Drug Analytics Summary (Top 5) ---")
print(drug_summary.head(5))

# -------------------------------------------------------------
# Step 5: Feature Extraction & Model Training
# -------------------------------------------------------------
X = df["Cleaned_Review"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(stop_words="english", max_features=300)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# -------------------------------------------------------------
# Step 6: Evaluation
# -------------------------------------------------------------
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("=" * 70)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["Negative", "Positive"]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=["Negative", "Positive"], zero_division=0))

# -------------------------------------------------------------
# Step 7: Predict Sentiment for New Patient Drug Reviews
# -------------------------------------------------------------
new_reviews = [
    "This medication brought fast relief from severe headache and fever with no side effects.",
    "Experienced terrible nausea, dizziness, and intense stomach pain after taking this pill."
]

cleaned_new = [clean_review_text(r) for r in new_reviews]
new_tfidf = vectorizer.transform(cleaned_new)
predictions = model.predict(new_tfidf)

print("\n--- Step 7: Clinical Sentiment Inference on New Reviews ---")
for review, sentiment in zip(new_reviews, predictions):
    print(f"\nReview    : \"{review}\"")
    print(f"Sentiment : {sentiment} Drug Feedback")

print("\n" + "=" * 70)
print("Experiment 7 execution completed successfully!")
print("=" * 70)
