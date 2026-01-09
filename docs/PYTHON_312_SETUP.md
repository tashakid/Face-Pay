# Python 3.12.9 Setup Guide

## 🐍 Creating Virtual Environment with Python 3.12.9

Since you have Python 3.12.9 installed, let's create a fresh virtual environment with this version to avoid dependency issues.

## 📋 Step-by-Step Instructions

### 1. Remove Old Virtual Environment (Optional)

If you want to start fresh:

```cmd
cd C:\Users\kinyu\payment-backend\face-payment-backend
rmdir /s venv
```

### 2. Create New Virtual Environment with Python 3.12.9

```cmd
cd C:\Users\kinyu\payment-backend\face-payment-backend
python -m venv venv
```

### 3. Activate the Virtual Environment

**For PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**For Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

### 4. Verify Python Version

```cmd
python --version
```

Should show: Python 3.12.9

### 5. Updated Requirements for Python 3.12.9

Create a new `requirements.txt` with these Python 3.12.9 compatible versions:

```txt
opencv-python
numpy>=1.26.4
mediapipe
firebase-admin
requests
python-dotenv
fastapi
uvicorn[standard]
pillow
pydantic>=2.0.0
python-jose[cryptography]
```

### 6. Install Dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔧 Updated Dependencies for Python 3.12.9

The requirements have been updated to ensure compatibility with Python 3.12.9:

- **numpy>=1.26.4**: Latest version compatible with Python 3.12
- **pydantic>=2.0.0**: Pydantic v2 for better Python 3.12 support
- **uvicorn[standard]**: Standard version with better performance
- **Other packages**: Latest versions that support Python 3.12

## 🚀 Alternative: Use Conda with Python 3.12

If you prefer using Conda:

```cmd
conda create -n face-payment python=3.12.9 -y
conda activate face-payment
pip install -r requirements.txt
```

## 🧪 Test the Setup

After installation, run the test script:

```cmd
python test_setup.py
```

## 📝 Benefits of Python 3.12.9

- **Better Performance**: Python 3.12 has significant performance improvements
- **Enhanced Error Messages**: More descriptive error messages
- **Improved Type Hints**: Better type hinting support
- **Faster Startup**: Reduced application startup time
- **Better Memory Management**: Improved memory efficiency

## 🔍 Troubleshooting

### If you encounter issues:

1. **Clear pip cache**:
   ```cmd
   pip cache purge
   ```

2. **Install packages individually**:
   ```cmd
   pip install numpy>=1.26.4
   pip install opencv-python
   pip install mediapipe
   pip install fastapi uvicorn[standard]
   pip install pillow pydantic>=2.0.0
   pip install python-jose[cryptography]
   pip install firebase-admin requests python-dotenv
   ```

3. **Use pre-compiled wheels**:
   ```cmd
   pip install --only-binary=all -r requirements.txt
   ```

## 🎯 Next Steps

Once the environment is set up:

1. **Configure environment variables** in `.env`
2. **Download SFace model** from OpenCV Zoo
3. **Run the application**:
   ```cmd
   python src/main.py
   ```

## 📚 Additional Resources

- [Python 3.12 Release Notes](https://docs.python.org/3.12/whatsnew/3.12.html)
- [OpenCV Python Installation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [MediaPipe Installation Guide](https://google.github.io/mediapipe/getting_started/install.html)

Python 3.12.9 should provide better performance and compatibility for the Face Recognition Payment System!