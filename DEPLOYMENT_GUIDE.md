# 🚀 LoanSight: University Presentation & Deployment Guide

This guide explains how to prepare, upload, and deploy your Loan Approval Prediction dashboard so that you can show it off in your university class!

---

## 🌟 Method A: Free Cloud Deployment (Highly Recommended)
Deploying your dashboard to the web is the **most professional approach**. It runs on Streamlit's cloud servers, meaning you don't need to install Python, TensorFlow, or Scikit-Learn on the classroom projector computer. You can just open a browser, click your custom link, and present.

### Step 1: Push Your Code to GitHub
1. Create a free account on [GitHub](https://github.com) if you don't have one.
2. Create a new repository named `loan-approval-prediction` (set it to **Public**).
3. Upload your project files. Make sure the repository contains:
   - `dashboard.py` (Streamlit frontend)
   - `train_models.py` (training pipeline script)
   - `loan_data.csv` (original dataset)
   - `requirements.txt` (the list of libraries we created for you)
   - `models/` directory (containing the default `.pkl` and `.keras` models so the app works immediately)
   - `Loan_Approval_Prediction.ipynb` (the interactive Jupyter Notebook)

### Step 2: Deploy to Streamlit Community Cloud (Free)
1. Go to the [Streamlit Community Cloud](https://share.streamlit.io) website.
2. Click **Connect with GitHub** and log in with your GitHub account.
3. Once logged in, click the **"New app"** button.
4. Fill in the repository details:
   - **Repository:** Select your `YOUR_GITHUB_USERNAME/loan-approval-prediction` repo.
   - **Branch:** Select `main` (or `master`).
   - **Main file path:** Enter the path to your dashboard file. If you put everything in the repository root, enter `dashboard.py`. If it is inside a folder, enter `Complete_Project/dashboard.py` (or similar).
5. Click **"Deploy!"**

Within 1–2 minutes, Streamlit will read your `requirements.txt`, install all libraries, and launch your app. 
You will get a public link (e.g., `https://your-app-name.streamlit.app`) that you can open on any projector, computer, or even mobile phone!

---

## 💻 Method B: Run Locally on the Classroom PC (Backup)
If the classroom PC is offline or you prefer running it locally, follow these steps to run the dashboard from your GitHub repository.

### Step 1: Install Python
Ensure Python (version 3.8 to 3.11 is recommended) is installed on the classroom computer. You can download it from [python.org](https://www.python.org/downloads/). Make sure to check **"Add Python to PATH"** during installation.

### Step 2: Download the Project
You can download your repository in two ways:
* **Option 1 (Easiest):** Go to your GitHub repository URL, click the green **Code** button, and select **"Download ZIP"**. Extract the ZIP file on the classroom computer.
* **Option 2 (Git):** Open Command Prompt / Terminal and run:
  ```bash
  git clone https://github.com/YOUR_GITHUB_USERNAME/loan-approval-prediction.git
  cd loan-approval-prediction
  ```

### Step 3: Install Dependencies
Open Command Prompt (Windows) or Terminal (macOS/Linux) in the folder where your files are extracted, and run:
```bash
pip install -r requirements.txt
```
*Note: This installs Streamlit, Scikit-Learn, TensorFlow, Joblib, Matplotlib, and Seaborn automatically.*

### Step 4: Launch the Dashboard
Run the following command in the Command Prompt:
```bash
streamlit run dashboard.py
```
Streamlit will launch locally and automatically open the web application in the default browser at **`http://localhost:8501`**.

---

## 🎓 Tips for a Successful University Presentation
1. **Highlight the Interactive Tuning Page:** Show the class how changing parameters (like Decision Tree depth or ANN epochs) live in the UI directly alters the confusion matrix and accuracy scores in real-time. This shows the professor that you understand hyperparameter relationships.
2. **Present the Word Report:** Have your generated [loan_prediction_report.docx](loan_prediction_report.docx) document ready on a USB drive or downloaded from your GitHub. Show the structured sections and the answers to the 20 questions as evidence of your research.
3. **Show the Interactive Notebook:** Open the [Loan_Approval_Prediction.ipynb](Loan_Approval_Prediction.ipynb) file in Jupyter and run the interactive ipywidgets form live. It shows you know how to build UI elements in both Jupyter Notebooks and Streamlit.
