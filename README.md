# 📚 BookMind — Personalized Book Recommendations with KNN

> A personalized book recommendation system built using K-Nearest Neighbors (KNN) collaborative filtering via the **Surprise** library.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![Surprise](https://img.shields.io/badge/Surprise-KNN-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo

| Interface | Description |
|-----------|-------------|
| 🌐 **Web UI** | Open `static/index.html` in any browser |
| 🎛️ **Streamlit App** | Run locally with `streamlit run app/streamlit_app.py` |
| 📓 **Notebook** | Open `notebooks/BookRecommender_KNN.ipynb` in Google Colab |

---

## 📌 Project Overview

This project demonstrates how to use the **Surprise** library to build a personalized book recommendation system. It uses the **GoodBooks-10K** dataset and implements three KNN-based collaborative filtering algorithms:

| Algorithm | Description | RMSE | MAE |
|-----------|-------------|------|-----|
| `KNNBasic` | Raw neighbor ratings, no bias correction | ~0.89 | ~0.70 |
| `KNNWithMeans` | Adjusts for user average rating bias | ~0.86 | ~0.68 |
| `KNNBaseline` ⭐ | Corrects global avg + user bias + item bias | ~0.83 | ~0.65 |

**KNNBaseline achieves the best performance.**

---

## 📂 Project Structure

```
book-recommender/
│
├── app/
│   └── streamlit_app.py        # Streamlit web application
│
├── static/
│   └── index.html              # Standalone web frontend (HTML/CSS/JS)
│
├── notebooks/
│   └── BookRecommender_KNN.ipynb   # Google Colab notebook
│
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/srinistha-biswas/book-recommender-knn.git
cd book-recommender-knn
```

### 2. Install dependencies
```bash
pip install numpy==1.26.4
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

### 4. Open the Web UI
Just open `static/index.html` in your browser — no server needed!

---

## 📊 Dataset

**GoodBooks-10K** by Zygmunt Zając  
- `books.csv` — metadata for 10,000 books (title, author, ratings)
- `ratings.csv` — ~6 million user ratings (user_id, book_id, rating 1–5)
- Source: [github.com/zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)

> **Note:** A sample of 10,000 ratings is used for training to ensure compatibility with free-tier compute environments like Google Colab.

---

## 🧠 How It Works

```
User Ratings → Cosine Similarity → KNN (k=10) → Predict Ratings → Top-N Recommendations
```

1. **Load ratings** from the dataset
2. **Compute cosine similarity** between users based on their rating history
3. **Find k=10 nearest neighbors** for the target user
4. **Aggregate neighbor ratings** to predict scores for unread books
5. **Return Top-N books** with the highest predicted ratings

---

## 📈 Features

- 🎯 **User-based recommendations** — Enter any user ID, get personalized book picks
- 🔍 **Item-based similar books** — Find books similar to any given book
- 📊 **Model comparison** — RMSE & MAE chart comparing all 3 KNN variants
- 🌐 **Web frontend** — Beautiful standalone HTML/CSS/JS UI
- 🎛️ **Streamlit dashboard** — Interactive Python-based web app

---

## 🏫 Academic Details

| Field | Details |
|-------|---------|
| **Name** | Srinistha Biswas |
| **Department** | CSE (IOTCSBT) |
| **Roll No.** | 9 |
| **Enrollment No.** | 12024052017026 |
| **Subject** | AIML Lab |
| **Tools** | Python, Surprise, Google Colab, Streamlit |

---

## 📄 License

MIT License — free to use, modify, and distribute.
