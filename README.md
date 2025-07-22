# 🩺 AI Health Assistant

This project is a simple **medical symptom checker** that predicts the most likely diseases based on user-selected symptoms. It uses a **Decision Tree Classifier** trained on a publicly available dataset and provides basic **precautionary measures**. The app is built with **Python, scikit-learn, and Streamlit**, and also lets users **download a PDF summary** of the results.

## 🚀 Key Features
- Predicts the **top 2 most probable diseases** based on chosen symptoms
- Displays a **confidence level** for each prediction
- Shows **precautions and recommended steps** for predicted diseases
- Generates a **downloadable PDF report** for the selected symptoms and outcomes
- Provides an **easy-to-use Streamlit web interface**

## 📂 Folder Structure
```
AIHealthAssistant/
│
├── app.py                  # Streamlit web app
├── train_model.py          # Script for training the model
├── model/                  # Folder where the trained model is saved after running train_model.py
│   └── dt_model.pkl        # Generated after training
├── data/
│   ├── disease_symptoms.csv
│   └── disease_precautions.csv
├── requirements.txt        # Dependencies list
├── README.md               # Documentation for the project
└── concepts.md             # Beginner-friendly explanation of ML concepts used
```

**Note:** The `model/` folder will be empty when you first clone the project. You must run:
```bash
python train_model.py
```
to train the classifier and create `model/dt_model.pkl` before using the app.

## 🗂 Dataset
- Dataset comes from a **publicly available source** (Kaggle)
- Contains **100+ symptoms** mapped to **40+ different diseases**
- Includes **precautionary suggestions** for each disease in a separate CSV

## ⚙️ Setup Instructions
```bash
# 1. Clone this repository
 git clone https://github.com/yourusername/AIHealthAssistant.git
 cd AIHealthAssistant

# 2. (Optional) Create a virtual environment
 python -m venv venv
 source venv/bin/activate   # macOS/Linux
 venv\\Scripts\\activate     # Windows

# 3. Install the required libraries
 pip install -r requirements.txt

# 4. Train the model
 python train_model.py

# 5. Launch the app
 streamlit run app.py
```

## 🖥️ How to Use
1. Pick one or more symptoms from the dropdown menu.
2. Click **Predict Disease**.
3. View the **top 2 predicted diseases**, their confidence levels, and suggested precautions.
4. Download a **PDF report** with all the details.

## 📊 Model & Evaluation
- Uses a **Decision Tree Classifier** for prediction
- Achieves around **90% accuracy** on test data
- Shows a **classification report** and **confusion matrix** during training for evaluation

## 🔧 Tools and Technologies
- **Python 3.10+**
- **scikit-learn** for building and training the ML model
- **Streamlit** for creating the web app interface
- **FPDF** for generating downloadable reports

## 📌 Possible Future Enhancements
- Improve accuracy with **Random Forest or Gradient Boosting models**
- Add **real-time medical APIs** for more details on diseases
- Enable **user accounts and history tracking**

## 🙌 Credits
- Dataset adapted from **Kaggle - Disease Symptom Prediction**
- Inspired by the idea of simple **AI-based health assistants**
