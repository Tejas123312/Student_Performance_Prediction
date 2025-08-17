# Student Performance Prediction

The **Student Performance Prediction** project leverages data-driven techniques and machine learning models to analyze and forecast students’ academic outcomes.  
It aims to predict performance and enable early intervention so educators can provide personalized support and improve overall learning outcomes.

---

## 🚀 Features

- 📊 Collects and processes student data (gender, ethnicity, parental education, lunch type, test preparation, reading/writing scores)
- 🤖 Uses machine learning algorithms to identify patterns and correlations
- 🔮 Predicts student math scores based on input features
- 🧑‍🏫 Helps teachers provide personalized guidance and support
- 📑 Generates detailed reports and actionable recommendations

---

## 🛠️ Tools & Technologies Used

- **Programming Language**: Python
- **Libraries/Frameworks**:
  - Pandas, NumPy (Data Preprocessing & Analysis)
  - Scikit-learn, CatBoost, XGBoost (Machine Learning Models)
  - Matplotlib, Seaborn (Data Visualization)
  - Flask (Web Application)
  - Dill (Serialization)
- **IDE/Editor**: Jupyter Notebook / VS Code
- **Dataset**: Historical student academic data (`notebook/data/stud.csv`)

---

## ⚙️ Workflow

1. **Data Collection**  
   - Student-related data (gender, ethnicity, parental education, lunch, test preparation, reading/writing scores)
2. **Data Preprocessing**  
   - Cleaning, handling missing values, normalization, encoding categorical features
3. **Exploratory Data Analysis (EDA)**  
   - Visualizing trends & feature correlations (`notebook/eda.ipynb`)
4. **Model Training & Evaluation**  
   - ML algorithms (Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, KNN, AdaBoost, CatBoost, XGBoost)
   - Model comparison and selection (`notebook/model_training.ipynb`)
5. **Prediction & Classification**  
   - Predicting student math scores via web interface
6. **Report Generation**  
   - Insights and recommendations using AI pipeline

---

## 📦 Project Structure

```
├── app.py / application.py         # Flask web application entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Project setup for packaging
├── src/                            # Source code
│   ├── __init__.py
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   ├── utils.py                    # Utility functions (model evaluation, object saving)
│   ├── components/                 # Data ingestion, transformation, model training modules
│   └── pipeline/                   # Prediction pipeline (CustomData, PredictPipeline)
├── artifacts/                      # Saved models, preprocessors, datasets
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   ├── test.csv
│   └── data.csv
├── notebook/
│   ├── eda.ipynb                   # Exploratory Data Analysis notebook
│   ├── model_training.ipynb        # Model training and evaluation notebook
│   └── data/
│       └── stud.csv                # Raw student data
├── templates/
│   ├── index.html                  # Home page
│   └── home.html                   # Prediction form and results
├── .ebextensions/                  # AWS Elastic Beanstalk config
│   └── python.config
├── catboost_info/                  # CatBoost training logs
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   └── learn/
│       └── events.out.tfevents
└── README.md
```

---

## 🌐 Web Application

- **Home Page**: `/`  
  Displays a welcome message.
- **Prediction Page**: `/predictdata`  
  Form for entering student details and scores.  
  On submission, predicts the math score using the trained model and displays the result.

### Example Usage

1. Run the Flask app:
    ```sh
    python app.py
    ```
2. Open your browser at `http://localhost:5000`
3. Enter student details and scores in the form.
4. View the predicted math score.

---

## 🧩 Model Training

- Data is split into train/test sets.
- Categorical features are one-hot encoded; numerical features are scaled.
- Multiple regression models are trained and evaluated.
- Best model is selected based on R2 score and saved as `model.pkl`.
- Preprocessing pipeline is saved as `preprocessor.pkl`.

---

## 📊 Model Performance

- Models compared: Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, KNN, AdaBoost, CatBoost, XGBoost
- Top R2 scores (test set):
    - Ridge: 0.88
    - Linear Regression: 0.88
    - Random Forest: 0.85
    - CatBoost: 0.85
    - AdaBoost: 0.85

---

## 📌 Future Enhancements

- Integration with a real-time student dashboard
- Deployment as a web application for schools and colleges
- Use of deep learning models for improved accuracy
- Personalized recommendation system for students

---

## 👨‍💻 Contributors

- Tejas Udgiri
- Adwait Palsule

---

## 📥 Installation

1. Clone the repository:
    ```sh
    git clone <repo-url>
    cd Student_Performance_Prediction
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the application:
    ```sh
    python app.py
    ```

---

## 📝 License

This project is for educational purposes.
