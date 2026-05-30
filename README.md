# Fake News Detection System

## Project Description

The Fake News Detection System is a machine learning-based web application designed to identify whether a news article is real or fake. The system uses Natural Language Processing (NLP) and machine learning algorithms to analyze news content and classify information based on textual patterns.

Users can either paste news content manually or provide a news article URL for verification. The application predicts results and displays confidence scores through a professional Flask-based web interface.

---

## Features

* Fake and Real News Detection
* News Text Analysis
* News URL Verification
* Confidence Score Prediction
* Analytics Dashboard
* Model Accuracy Comparison
* Confusion Matrix Visualization
* Professional Dark-Themed Flask UI

---

## Technologies Used

* Python
* Flask
* Machine Learning
* Natural Language Processing (NLP)
* Scikit-learn
* Pandas
* NumPy
* BeautifulSoup
* Newspaper3k
* HTML
* CSS

---

## Machine Learning Models Used

* Passive Aggressive Classifier
* Logistic Regression
* Naive Bayes
* Random Forest

The system automatically selects the best-performing model based on accuracy.

---

## Dataset

Dataset used:

* `Fake.csv`
* `True.csv`

The dataset is used for training and testing the machine learning model for fake news classification.

---

## Project Structure

```text
Fake-News-Detection/
│── app.py
│── train_model.py
│── preprocess.py
│── requirements.txt
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── style.css
│   ├── accuracy_chart.png
│   └── confusion_matrix.png
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/krishnagoutham2004-afk/Fake-News-Detection-System.git
```

### 2. Navigate to Project Folder

```bash
cd Fake-News-Detection
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train_model.py
```

### 5. Run Flask Application

```bash
python app.py
```

### 6. Open in Browser

```text
http://127.0.0.1:5000
```

---

## Screenshots

Add screenshots of:

* Homepage UI
  <img width="1208" height="1965" alt="homepage" src="https://github.com/user-attachments/assets/1b5979c4-a15b-4595-8652-eb5df7632ed7" />

* Prediction Result Page
  <img width="1208" height="1115" alt="prediction" src="https://github.com/user-attachments/assets/6e9bfec0-bac5-48d9-8de9-d572d0367a3f" />
---

## Future Improvements

* Better real-time news verification
* Improved prediction accuracy
* Multilingual support
* API integration for live news analysis

---

## Author

**Goutham Krishna**
