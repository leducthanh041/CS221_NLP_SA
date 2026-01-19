# 🛡️ Vietnamese Hate Speech Detection

> **Course Project:** Natural Language Processing (NLP)
> **Topic:** Research and development of models for classifying toxic comments on Vietnamese social media.

## 👥 Team Members

| No. | Full Name | Student ID | Role |
| --- | --- | --- | --- |
| 1 | **Le Duc Thanh** | 23521441 | Team Leader, EDA, Preprocessing, Deployment, Report Writing |
| 2 | **Nguyen Ngoc Thanh Phuong** | 23521245 | Modeling, Report Writing |
| 3 | **Huynh Dien Thuc** | 23521555 | Modeling, Report Writing |

**👨‍🏫 Supervisor:** Nguyen Trong Chinh. PhD

---

## 📖 Project Overview

This project focuses on solving the **Multi-class Text Classification** problem for the Vietnamese language, aiming to automatically detect content that violates community standards. We utilize the standard **UIT-ViHSD** dataset to train and evaluate machine learning models ranging from basic to advanced architectures.

The system classifies data into 3 labels:

* 🟢 **CLEAN (0):** Clean, civilized, and constructive comments.
* 🟡 **OFFENSIVE (1):** Vulgar or slightly offensive comments, but without hate speech or incitement.
* 🔴 **HATE (2):** Hate speech, containing heavy personal or organizational attacks.

---

## 🛠️ Methodology

To address this problem, the team applied a rigorous data processing and feature extraction workflow:

### 1. Preprocessing

The text cleaning steps are designed specifically for the characteristics of Vietnamese social media language:

1. **Word Segmentation:** Using specialized libraries to handle Vietnamese compound words (e.g., combining single syllables like `đất` and `nước` into `đất_nước`).
2. **Safe Stopword Removal:** Removing stop words but retaining the original sentence if filtering removes all content (preventing data loss).
3. **Normalization:** Lowercasing text, removing URLs, and Mentions (@user).
4. **Regex Cleaning:** Handling Hashtags (removing `#` but keeping the content) and preserving important Emojis that carry emotional nuance.

### 2. Feature Extraction

We utilize **TF-IDF** combined with **N-grams (Unigram & Bigram)** to convert text into numerical vectors. This method helps the model effectively capture characteristic offensive phrases such as `lũ_ngu` (bunch of idiots), `vô_học` (uneducated).

### 3. Modeling

The team conducted experiments and compared the effectiveness of two approaches:

* **Traditional Machine Learning:** Naive Bayes (Multinomial), Logistic Regression, SVM, KNN, Decision Tree.
* **Deep Learning:** Fine-tuning the pre-trained language model **PhoBERT**.

---

## 📊 Experimental Results

Evaluation results on the independent **Test set**:

| Model | Accuracy | F1-Score (Macro) | Remarks |
| --- | --- | --- | --- |
| **PhoBERT** | **86.78%** | **63.16%** | Best performance (State-of-the-art) due to context understanding. |
| **Logistic Regression** | 86.33% | 62.80% | Very good, approximating PhoBERT's performance. |
| **SVM** | 86.11% | 62.17% | Stable and effective. |
| **Naive Bayes (MNB)** | 84.56% | 54.11% | Very fast inference speed, decent accuracy. |
| **KNN** | 83.61% | 46.08% | Lowest performance. |
| **Decision Tree** | 82.62% | 51.66% | Prone to overfitting. |

---

## 🚀 Deployment (Demo App)

To illustrate the research results, the team built a lightweight Web application using **Streamlit**, utilizing the **Naive Bayes** and **SVM** models. This is a compact version, optimized for response speed for practical deployment.

### Demo Folder Structure

The source code for the demo application is located in the `uit_vihsd_streamlit_demo` directory:

```text
uit_vihsd_streamlit_demo
├── README copy.md                      # Backup document
├── README.md                           # App instructions
├── app.py                              # Streamlit interface source code
├── final_best_mnb_tfidf.joblib         # Trained Naive Bayes Model
├── final_best_mnb_tfidf_info.json      # Metadata
├── navie_bayes.ipynb                   # Testing Notebook
├── requirements.txt                    # Required libraries for the App
├── svm_best.joblib                     # Trained SVM Model
├── utils.py                            # Shared preprocessing module
└── vietnamese-stopwords.txt            # Vietnamese stopword list

```

### How to Run the Demo

If you want to experience the quick classification app:

1. Navigate to the demo folder: `cd uit_vihsd_streamlit_demo`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

---

## 🤝 Acknowledgements

We would like to express our sincere gratitude to **Dr. Nguyen Trong Chinh** for his dedicated guidance and valuable feedback that helped the team complete this project.

---

*© 2024 NLP Team PhuongThanhThuc. Project developed for Natural Language Processing.*