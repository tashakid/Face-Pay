# How to Activate Your Python 3.12 Virtual Environment

## ✅ Virtual Environment Created Successfully!

I've created a new virtual environment using Python 3.12 specifically. Here's how to activate and use it:

## 🔄 Activation Instructions

### Option 1: PowerShell (Recommended)

1. **Open PowerShell**
2. **Navigate to project directory**:
   ```powershell
   cd C:\Users\kinyu\payment-backend\face-payment-backend
   ```

3. **Set Execution Policy** (if you haven't already):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   Press `Y` when prompted.

4. **Activate the virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

### Option 2: Command Prompt (cmd)

1. **Open Command Prompt**
2. **Navigate to project directory**:
   ```cmd
   cd C:\Users\kinyu\payment-backend\face-payment-backend
   ```

3. **Activate the virtual environment**:
   ```cmd
   venv\Scripts\activate.bat
   ```

## ✅ Verify Python Version

After activation, verify you're using Python 3.12:

```cmd
python --version
```

Should show: `Python 3.12.x`

## 📦 Install Dependencies

Once activated, install the required packages:

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

## 🧪 Test the Setup

Run the test script to verify everything works:

```cmd
python test_setup.py
```

## 🎯 What We Did

- Used `py -3.12 -m venv venv` to create the virtual environment with Python 3.12
- This ensures the venv uses Python 3.12 instead of the default Python 3.15
- Updated requirements.txt with Python 3.12 compatible packages

## 🔍 Quick Verification

You should see `(venv)` at the beginning of your prompt after activation:

```
(venv) C:\Users\kinyu\payment-backend\face-payment-backend>
```

## 🚀 Next Steps

1. Activate the environment using one of the methods above
2. Verify Python version shows 3.12
3. Install dependencies with `pip install -r requirements.txt`
4. Test with `python test_setup.py`
5. Run the application with `python src/main.py`

Your Python 3.12 environment is now ready! 🎉