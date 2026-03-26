# 📧 Spam Email Classifier using Machine Learning  
A complete end-to-end project that builds an ML model to classify emails as **Spam** or **Ham (Not Spam)** using Scikit-Learn.

<img width="1536" height="1024" alt="Spam email classifier workflow illustration" src="https://github.com/user-attachments/assets/6b034b3b-0565-4173-87fd-3fe053cd16af" />

---

## 🚀 Project Overview
This project builds a **Machine Learning Email Spam Detector** using:
- **TF-IDF Vectorizer**
- Evaluation with accuracy, precision, recall & confusion matrix
  

  ## 🔥 Project Highlights

- 🧹 Clean & preprocess text data  
- 🔤 Convert text to numeric features using **TF-IDF**  
- 🤖 Build & train a **Logistic Regression** model  
- 📊 Evaluate model with industry-grade metrics  
- 📨 Predict whether new messages are spam  
- 📓 Includes a ready-to-run **Jupyter Notebook**


---

## 📁 Repository Structure
```
📂 Spam-Email-Classifier-ML
│── README.md
│── spam_classifier.ipynb
│── requirements.txt
│
│  
│── 📂 models
│     ├── vectorizer.pkl
│     ├── classifier.pkl
│── 📂 assets
│     ├── data_preview.png
│     ├── Label count.png
│     ├── model accuracy.png
│     ├── TF-IDF.png
│     ├── classification report.png
│     ├── prediction_demo.png
│── 📂 src
      ├── spam_classifier.py

```

---

## 🧠 Technologies Used
- Python  
- Scikit-Learn  
- Pandas  

---

🤖 Model Workflow

1. Load dataset
2. Encode labels
3. Split training/testing data
4. TF-IDF vectorization
5. Train logistic regression
6. Evaluate model
7. Predict spam/ham

## 🧹 Data Preprocessing
The following steps were performed:

✔ Remove special characters  
✔ Convert to lowercase  
✔ Remove stopwords  
✔ Lemmatize text  
✔ Apply TF-IDF vectorization  

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stopwords]
    return " ".join(words)
```

---

## 🏗️ Model Training
## 🔤 TF-IDF Vectorization
Convert messages into numerical format using TF-IDF.
```python
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF Train Shape:", X_train_tfidf.shape)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
```

---

## 📊 Model Evaluation
Checking accuracy and classification metrics.
```python
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
```

### 🔍 Sample Metrics
- Accuracy: **97%**
- Precision: **0.96**
- Recall: **0.95**
- F1 Score: **0.95**

---

## 🖼️ Screenshots

### 📌 Dataset Preview  
<img width="670" height="266" alt="dataset_output_info" src="https://github.com/user-attachments/assets/f17decd0-d0f2-4daa-ae9e-07a2e3e37140" />


### 📌 Label Count Info  
<img width="662" height="151" alt="Label_count_info" src="https://github.com/user-attachments/assets/20f31655-7fa3-4b92-8ac4-16f59ee8d8ab" />


### 📌 Model Accuracy Output
<img width="661" height="119" alt="Model_accuracy_output" src="https://github.com/user-attachments/assets/029e0371-149d-4df0-9c09-c9853c227031" />

### 📌 TF-IDF Output
<img width="661" height="119" alt="Model_accuracy_output" src="https://github.com/user-attachments/assets/a5458664-5ccb-4f82-923c-22903e16c953" />


### 📌 Classification Report  
<img width="681" height="245" alt="Classification_report_output" src="https://github.com/user-attachments/assets/54aab2f1-b6ab-4701-94b9-1df553a9a843" />


### 📌 Prediction Demo  

<img width="663" height="164" alt="spam_output" src="https://github.com/user-attachments/assets/287f96c0-eebf-4323-8eaf-a09a7ba7ca9b" />


---

## ▶️ How to Run the Project

### **1️⃣ Clone the repo**
```bash
git clone https://github.com/your-username/Spam-Email-Classifier.git
cd Spam-Email-Classifier
```

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run training**
```bash
python src/train.py
```

### **4️⃣ Test single prediction**
```bash
python src/predict.py ["Congratulations! You've won a free iPhone. Click here now!"]
```

---

## 📦 Saved Models
After training, these files are generated:
- `vectorizer.pkl`
- `classifier.pkl`

You can import them for reuse anywhere.

---

## ⭐ Future Enhancements
- Add deep learning (LSTM / BERT)
- Deploy using Flask or FastAPI
- Create Streamlit Web App
- Add real-time email scanning

---

## 💬 Contribute
Pull requests are welcome!  
If you like this project, **star ⭐ the repository**.

---
🏅 Author

👤 Your Name : Gauri Giri
📧 Email: gaurigiri777@gmail.com
🔗 GitHub: https://github.com/yourusername
