import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = {
    "plot": [
        "A police officer fights criminals to protect the city",
        "A superhero saves the world from an alien invasion",
        "A brave soldier goes to war to defend his country",
        "Two friends go on a hilarious road trip",
        "People face funny situations in daily life",
        "A man struggles with love and heartbreak",
        "A couple falls in love despite family opposition",
        "A family deals with emotional challenges",
        "A young girl overcomes depression",
        "Scientists discover a new planet",
        "A time traveler changes the future",
        "Robots take control of the earth"
    ],
    "genre": [
        "Action","Action","Action",
        "Comedy","Comedy",
        "Romance","Romance",
        "Drama","Drama",
        "Sci-Fi","Sci-Fi","Sci-Fi"
    ]
}

df = pd.DataFrame(data)

X = df["plot"]
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_plot = ["A scientist invents a machine to control time"]
new_plot_tfidf = tfidf.transform(new_plot)
print("Predicted Genre:", model.predict(new_plot_tfidf)[0])