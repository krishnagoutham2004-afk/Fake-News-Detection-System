from flask import Flask, render_template, request
import joblib
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from preprocess import preprocess_text

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def extract_article_text(url):
    """
    Extract article text from URL
    using newspaper3k and BeautifulSoup.
    """

    try:
        article = Article(url)
        article.download()
        article.parse()

        text = article.text

        # If extraction is good
        if len(text.strip()) > 200:
            return text

    except:
        pass

    try:
        headers = {
            "User-Agent":
            "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        text = " ".join(
            [p.get_text()
             for p in paragraphs]
        )

        return text

    except:
        return ""


@app.route('/')
def home():
    return render_template(
        'index.html'
    )


@app.route('/predict', methods=['POST'])
def predict():

    news_text = request.form.get(
        'news'
    )

    news_url = request.form.get(
        'url'
    )

    extracted_text = ""

    # URL input
    if news_url:

        extracted_text = (
            extract_article_text(
                news_url
            )
        )

    # Text input
    elif news_text:

        extracted_text = (
            news_text
        )

    else:

        return render_template(
            'result.html',
            prediction=
            "No Input Provided ❌",
            confidence=0,
            status="fake",
            accuracy=99.73
        )

    # Extraction failed
    if len(
        extracted_text.strip()
    ) < 50:

        return render_template(
            'result.html',
            prediction=
            "Could not extract article ❌",
            confidence=0,
            status="fake",
            accuracy=99.73
        )

    # Preprocess text
    cleaned_news = preprocess_text(
        extracted_text
    )

    # Convert to vector
    vectorized_news = (
        vectorizer.transform(
            [cleaned_news]
        )
    )

    # Prediction
    prediction = model.predict(
        vectorized_news
    )[0]

    # Dynamic confidence score
    try:

        # Models with probability
        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    vectorized_news
                )[0]
            )

            confidence = (
                max(probabilities)
                * 100
            )

        # Models with decision function
        elif hasattr(
            model,
            "decision_function"
        ):

            score = abs(
                model.decision_function(
                    vectorized_news
                )[0]
            )

            confidence = min(
                score * 20,
                99.99
            )

        else:
            confidence = 75

    except:
        confidence = 75

    # Result
    if prediction == 0:

        result = (
            "Fake News ❌"
        )

        status = "fake"

    else:

        result = (
            "Real News ✅"
        )

        status = "real"

    return render_template(
        'result.html',
        prediction=result,
        confidence=round(
            confidence,
            2
        ),
        status=status,
        accuracy=99.73
    )


if __name__ == '__main__':
    app.run(debug=True)