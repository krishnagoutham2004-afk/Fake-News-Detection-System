import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    LogisticRegression
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess_text


# Create folders
os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)


# Load datasets
print("Loading dataset...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")


# Labels
fake["label"] = 0
true["label"] = 1


# Combine datasets
data = pd.concat(
    [fake, true],
    axis=0
)

# Shuffle
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Combine title + text
print("Combining title and text...")

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

data["content"] = (
    data["title"] +
    " " +
    data["text"]
)

# Keep required columns
data = data[
    ["content", "label"]
]


# Preprocess text
print("Preprocessing text...")

data["content"] = data[
    "content"
].apply(preprocess_text)


# Features and labels
X = data["content"]
y = data["label"]


# TF-IDF
print("Creating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Models
models = {

    "Passive Aggressive":
    PassiveAggressiveClassifier(
        max_iter=1000
    ),

    "Logistic Regression":
    LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes":
    MultinomialNB(),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


accuracies = {}

best_model = None
best_accuracy = 0
best_predictions = None


# Train models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    accuracies[name] = (
        accuracy * 100
    )

    print(
        f"{name} Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # Save best model
    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_predictions = predictions


# Save best model
print("\nSaving best model...")

joblib.dump(
    best_model,
    "model/fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)

print("Model saved successfully!")


# Accuracy chart
print("\nGenerating charts...")

plt.figure(figsize=(8, 5))

plt.bar(
    accuracies.keys(),
    accuracies.values()
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylabel("Accuracy (%)")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "static/accuracy_chart.png"
)

plt.close()


# Confusion matrix
cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "static/confusion_matrix.png"
)

plt.close()


print("\nDone!")
print(
    f"Best Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)
print(
    "Charts saved in static folder."
)