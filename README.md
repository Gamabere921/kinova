# ♻️ SpaceTrash Classifier

### *Revolutionizing Recycling on Mars* 🚀

This project is a simple web application designed to **classify waste materials from uploaded images** using a machine learning model.

The idea is to make **recycling and reuse easier** — even on **Mars**, where efficient resource management is crucial.

Originally developed for the **SpaceTrash Hack: Revolutionizing Recycling on Mars** challenge, this project demonstrates how computer vision and AI can support sustainable missions beyond Earth.

---

## 🌍 Overview

The web app allows users to:

- Upload an image of waste material (e.g., plastic, metal, glass, paper).
- Automatically detect and classify what type of material it is.
- Use that information to decide whether it can be recycled or reused.

---

## 🧠 Machine Learning Model

> ⚠️ The trained ML model used in the original implementation is not included in this repository.
> 
> 
> To enable material classification, you’ll need to integrate your own **trained model** (e.g., TensorFlow, PyTorch, or scikit-learn).
> 

Suggested approach:

- Train a CNN model to classify images into categories like *plastic, metal, glass, paper,* etc.
- Export the model and load it into the FastAPI app for inference.

---

## 🛠️ Installation

1. **Clone this repository**
    
    ```bash
    git clone <your-repository-url>
    cd <your-project-folder>
    
    ```
    
2. **Create and activate a Python virtual environment**
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows
    
    ```
    
3. **Install dependencies**
    
    ```bash
    pip install uvicorn fastapi
    
    ```
    

---

## 🚀 Running the App

To start the development server:

```bash
uvicorn web_app.app:app --host 0.0.0.0 --port 8000 --reload

```

Then open your browser and visit:

👉 [http://localhost:8000](http://localhost:8000/)

---

## 🧩 Project Structure

```
.
├── web_app/
│   ├── app.py           # Main FastAPI app
│   ├── static/          # Static assets (CSS, JS, images)
│   ├── templates/       # HTML templates
│   └── model/           # (Optional) Folder for your trained model
└── README.md

```

---

## 💡 Future Improvements

- Integrate a **real trained ML model** for live material detection.
- Add **data visualization dashboards** for recycling insights.
- Support **offline inference** for Mars base operations 😉.

---

## 🪐 Inspiration

> As humanity prepares for interplanetary exploration, waste management becomes a new frontier.
> 
> 
> This project aims to **promote sustainability — from Earth to Mars** — through intelligent automation and creative technology.
> 

---

## 🧑‍🚀 Author

Developed by **Gambare Mx**

For the **SpaceTrash Hack: Revolutionizing Recycling on Mars** 🚀
