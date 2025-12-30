# Environment Setup Instructions

## 🚨 Important: Windows Setup Required

Due to Windows compilation issues with NumPy and OpenCV, please follow these manual setup instructions.

## 📋 Prerequisites

1. **Python 3.8+** (Already detected: Python 3.15.0)
2. **Visual Studio Build Tools** (for compiling packages)
3. **Git** (for version control)

## 🔧 Setup Options

### Option 1: Using Conda (Recommended)

1. **Install Miniconda:**
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Choose Python 3.9 version for Windows
   - Install with default settings

2. **Create Conda Environment:**
   ```cmd
   conda create -n face-payment python=3.9 -y
   conda activate face-payment
   ```

3. **Install Core Packages with Conda:**
   ```cmd
   conda install numpy opencv mediapipe requests -y
   ```

4. **Install Remaining Packages with Pip:**
   ```cmd
   pip install fastapi uvicorn pillow pydantic python-jose[cryptography] python-dotenv firebase-admin
   ```

### Option 2: Using Pre-compiled Wheels

1. **Download Pre-compiled Wheels:**
   - Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/
   - Download these files:
     - numpy‑1.24.3‑cp39‑cp39‑win_amd64.whl
     - opencv_python‑4.8.0.76‑cp39‑cp39‑win_amd64.whl
     - mediapipe‑0.10.7‑cp39‑cp39‑win_amd64.whl

2. **Install Wheels:**
   ```cmd
   cd face-payment-backend
   venv\Scripts\activate
   pip install numpy‑1.24.3‑cp39‑cp39‑win_amd64.whl
   pip install opencv_python‑4.8.0.76‑cp39‑cp39‑win_amd64.whl
   pip install mediapipe‑0.10.7‑cp39‑cp39‑win_amd64.whl
   pip install fastapi uvicorn pillow pydantic python-jose[cryptography] python-dotenv firebase-admin
   ```

### Option 3: Install Visual Studio Build Tools

1. **Install Visual Studio Build Tools:**
   - Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - Select "C++ build tools" during installation

2. **Install Packages Normally:**
   ```cmd
   cd face-payment-backend
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

After successful setup:

1. **Activate Environment:**
   ```cmd
   # For venv:
   cd face-payment-backend
   venv\Scripts\activate
   
   # For conda:
   conda activate face-payment
   ```

2. **Set Up Environment Variables:**
   ```cmd
   copy .env.example .env
   # Edit .env file with your actual credentials
   ```

3. **Download SFace Model:**
   - Download from: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
   - Place `face_recognition_sface_2021dec.onnx` in the project root

4. **Run the Application:**
   ```cmd
   python src/main.py
   ```

## 📁 Project Structure

```
face-payment-backend/
├── venv/                    # Virtual environment
├── src/
│   ├── __init__.py
│   ├── main.py              # Main application
│   ├── vision.py            # Face detection & webcam
│   ├── auth.py              # SFace authentication
│   └── payment.py           # M-Pesa integration
├── .env                     # Environment variables
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## 🐛 Troubleshooting

### Common Issues:

1. **NumPy Installation Error:**
   - Solution: Use Conda or pre-compiled wheels

2. **OpenCV Installation Error:**
   - Solution: Install with Conda or download pre-compiled wheel

3. **MediaPipe Installation Error:**
   - Solution: Use Conda or specific version wheel

4. **Camera Not Working:**
   - Check camera permissions in Windows settings
   - Ensure no other app is using the camera

5. **SFace Model Not Found:**
   - Download the ONNX model file
   - Place it in the project root directory

### Verification Commands:

```cmd
# Check Python version
python --version

# Check installed packages
pip list

# Test OpenCV
python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Test MediaPipe
python -c "import mediapipe; print('MediaPipe version:', mediapipe.__version__)"
```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Ensure all prerequisites are installed
3. Try different setup options
4. Verify camera permissions
5. Check model file placement

## 🎯 Next Steps

Once environment is set up:

1. Configure your `.env` file with M-Pesa and Firebase credentials
2. Download the SFace model
3. Test the application with demo mode
4. Run full camera workflow