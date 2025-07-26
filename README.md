# 📰 Fake News Detection Using Machine Learning

This project classifies news headlines and articles as **Fake** or **Real** using machine learning techniques. It includes a simple interactive web interface built with **Streamlit** and also uses **LIME** for model explainability.

---

## 🚀 Features
- Logistic Regression, SVM, Random Forest Classifiers
- Data pre-processing using NLTK & TF-IDF
- LIME-based explanation of predictions
- User-friendly Streamlit app

---

## 🧠 Technologies Used
- Python
- Pandas, Numpy
- Scikit-learn
- NLTK
- LIME
- Streamlit

---

## 📁 Project Structure
| File/Folder | Description |
|-------------|-------------|
| `app.py` | Streamlit app |
| `phase3_preprocessing.ipynb` | Data cleaning & TF-IDF |
| `news_classifier.ipynb` | Model training |
| `phase7_lime_explainability.ipynb` | LIME explanation |
| `requirements.txt` | Dependencies |

---

## 💻 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
