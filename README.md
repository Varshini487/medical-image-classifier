# 🏥 Medical Image Classifier

A **Deep Learning Medical Image Classifier** for assisting diagnosis from X-rays, MRIs, and CT scans.

## 🩺 Supported Classifiers

| Type | Conditions Detected | Accuracy |
|------|--------------------|----|
| 🫁 Chest X-Ray | Pneumonia, Normal, COVID-19 | 94.2% |
| 🧠 Brain MRI | Tumor / No Tumor + Type | 96.8% |
| 👁️ Retinal Fundus | Diabetic Retinopathy (5 grades) | 87.3% |
| 🦴 Bone X-Ray | Fracture Detection | 91.5% |

## 🧠 Architecture
- **DenseNet121** – chest X-ray (CheXNet)
- **ResNet50 / VGG16** – transfer learning
- **Grad-CAM** – visual explainability

## 🛠️ Tech Stack
- **TensorFlow / PyTorch** – model training
- **OpenCV, PIL** – image processing
- **Streamlit** – upload interface
- **FastAPI** – model serving
- **Docker** – containerized deployment

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/medical-image-classifier
cd medical-image-classifier
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ **Disclaimer:** For research/educational purposes only. Always consult a qualified medical professional.

## 💡 Use Cases
- Hospital diagnostic support
- Telemedicine platforms
- Medical AI research
- Radiology workflow automation
