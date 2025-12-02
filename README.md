# 🌸 Flower Detection with Neural Networks

A machine learning project for flower classification using transfer learning and TensorFlow.js for web deployment.

## 🚀 Features

- **Transfer Learning**: Uses pre-trained neural networks for efficient flower classification
- **Web Interface**: Interactive web application for real-time flower detection
- **Multiple Classes**: Supports classification of various flower types
- **Easy Deployment**: Ready-to-use web model for browser-based inference

## 📁 Project Structure

```
├── flower_detection.ipynb      # Main training notebook
├── train_flowers_tl.py        # Training script with transfer learning
├── index.html                 # Web interface for flower detection
├── web_model_flowers/         # TensorFlow.js model files
├── flowers_class_names.js     # Class labels for web model
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Setup & Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FlowerDetection.git
cd FlowerDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Training the Model
Run the training script or use the Jupyter notebook:
```bash
python train_flowers_tl.py
```
Or open `flower_detection.ipynb` in Jupyter Lab/Notebook.

### Web Interface
1. Start a local server:
```bash
python -m http.server 8080
```

2. Open your browser and go to `http://localhost:8080`

3. Upload an image or use your camera to detect flowers!

## 🎯 Model Performance

The model uses transfer learning with a pre-trained convolutional neural network, achieving high accuracy on flower classification tasks.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- TensorFlow.js team for the web deployment framework
- The dataset providers for training data
- Transfer learning research community
