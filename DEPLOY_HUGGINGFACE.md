# Hugging Face Spaces Deployment Guide — EV Battery Failure Classifier

This guide provides step-by-step instructions to deploy the **EV Battery Failure Classification Streamlit Web App** to **Hugging Face Spaces** for free, instant cloud hosting.

---

## Method 1: Web Browser Drag & Drop (Simplest & Fastest - 2 Minutes)

### Step 1: Create a New Hugging Face Space
1. Log in to [Hugging Face](https://huggingface.co/). (Create a free account if you don't have one).
2. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
3. Fill in the Space settings:
   - **Space Name**: `ev-battery-failure-classifier` (or your preferred name)
   - **License**: `mit` (or Apache 2.0)
   - **Select the Space SDK**: Choose **Streamlit**
   - **Space Hardware**: Select **CPU Basic (Free - 2 vCPU, 16GB RAM)**
   - **Visibility**: Public (or Private)
4. Click **Create Space**.

### Step 2: Upload Files directly in the Browser
1. In your newly created Space page, click the **Files and versions** tab.
2. Click **Add file** -> **Upload files**.
3. Drag and drop the following files from your `p2` folder:
   - `app.py`
   - `model_loader.py`
   - `best_ev_battery_model.pkl`
   - `ev_battery_health_subset.csv`
   - `requirements.txt`
   - `README.md`
   - `.streamlit/config.toml` (create folder `.streamlit` and upload `config.toml`)
4. In the commit message box at the bottom, type `Initial app deployment` and click **Commit changes to main**.

### Step 3: Monitor Build & Launch
- Hugging Face will automatically detect `requirements.txt` and `app.py`.
- You will see the status change from **Building** -> **Running**.
- Once **Running** appears, your Streamlit app is live on Hugging Face!

---

## Method 2: Git CLI Deployment (Command Line Push)

If you prefer using Git in your local terminal:

### Step 1: Initialize Git in your project folder (if not done)
```bash
cd c:\Users\adity\OneDrive\Desktop\p2
git init
git add .
git commit -m "Initial commit"
```

### Step 2: Add Hugging Face Space as Git Remote
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

### Step 3: Push to Hugging Face
```bash
git push -u space main
```
*(Enter your Hugging Face username and Access Token when prompted for credentials).*

---

## Method 3: Hugging Face CLI Deployment

Using the official Hugging Face Python library:

```bash
# 1. Install Hugging Face Hub library
pip install huggingface_hub

# 2. Login to Hugging Face
huggingface-cli login

# 3. Upload folder directly
python -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='.', repo_id='YOUR_USERNAME/YOUR_SPACE_NAME', repo_type='space')"
```

---

## Required Files Checklist for Hugging Face Spaces

| File | Purpose | Required |
| :--- | :--- | :--- |
| `app.py` | Main Streamlit application entrypoint | Yes |
| `model_loader.py` | ML model loading & feature preprocessor | Yes |
| `best_ev_battery_model.pkl` | Tuned Logistic Regression model binary | Yes |
| `ev_battery_health_subset.csv` | Reference dataset for scaling statistics | Yes |
| `requirements.txt` | Dependency list (`streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `joblib`) | Yes |
| `README.md` | YAML metadata frontmatter specifying Streamlit SDK | Yes |
| `.streamlit/config.toml` | Production dark theme styling | Recommended |
