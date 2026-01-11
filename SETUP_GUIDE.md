# Complete Local Setup Guide

## Before You Start

**Important Prerequisites:**
- Python 3.11 or 3.12 (recommended: 3.12)
- Node.js 18+ and npm
- Git
- A GitHub account
- ngrok account (free tier is fine)
- Firebase project set up
- M-Pesa Daraja developer account (Sandbox)

**Estimated time to complete:** 30-45 minutes

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/tashakid/Face-Pay.git
cd Face-Pay
```

---

## Step 2: Set Up Python Backend

### 2.1 Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR for Windows: venv\Scripts\activate
```

**Verify activation:** Your terminal should show `(venv)` at the beginning of the prompt.

### 2.2 Install Python Dependencies (CRITICAL ORDER!)

**Follow this exact order to avoid dependency conflicts:**

```bash
# Step 1: Install base requirements
pip install --upgrade pip
pip install -r requirements.txt
```

**After the above completes, install the specific versions manually:**

```bash
# Step 2: Install NumPy FIRST (NumPy < 2.0 is CRITICAL)
pip install "numpy<2.0"

# Step 3: Install TensorFlow-CPU
pip install "tensorflow-cpu==2.17.1"

# Step 4: Install tf-keras adapter
pip install "tf-keras==2.16.0"
```

**Why this order?**
- `pip install -r requirements.txt` may try to upgrade NumPy to 2.0, which breaks OpenCV
- We explicitly install NumPy < 2.0 after requirements to ensure compatibility
- TensorFlow requires specific versions of NumPy and tf-keras

### 2.3 Verify Installation

```bash
python -c "import cv2; import deepface; import tensorflow; import numpy; print('All dependencies installed correctly!')"
```

If you see "All dependencies installed correctly!" you're good. If there are errors, see the Troubleshooting section below.

---

## Step 3: Firebase Setup

### 3.1 Create or Access Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Either create a new project or use an existing one
3. Navigate to **Project Settings** (gear icon)
4. Go to **Service accounts** tab
5. Click **Generate new private key**
6. Download the JSON file (it will have a name like `your-project-id-abc123.json`)

### 3.2 Place the Firebase Credentials

```bash
# Move the downloaded JSON file to the project root
mv /path/to/your-project-id-abc123.json ~/projects/Face-Pay/
```

**Security note:** Never commit this file to Git! It gives full access to your Firebase project.

---

## Step 4: M-Pesa Daraja Setup

### 4.1 Access M-Pesa Daraja Portal

1. Go to [Safaricom Developer Portal](https://developer.safaricom.co.ke/)
2. Sign in or create an account
3. Create a new app or use existing Sandbox app
4. You'll need:
   - **Consumer Key**
   - **Consumer Secret**
   - **PassKey** (for STK Push)
   - **Shortcode** (usually `174379` for Sandbox)

### 4.2 Note Down Credentials

Keep these handy, you'll add them to `.env` in Step 5.

---

## Step 5: Environment Variables Setup

### 5.1 Create .env File

```bash
cp .env.example .env
```

### 5.2 Edit .env File

```bash
nano .env  # or use any text editor
```

Replace with your actual values:

```env
# Firebase Credentials (absolute path to your downloaded JSON file)
FIREBASE_CREDENTIALS=/home/your-username/projects/Face-Pay/your-project-id-abc123.json

# M-Pesa Sandbox Credentials
MPESA_KEY=your_consumer_key_here
MPESA_SECRET=your_consumer_secret_here
MPESA_PASSKEY=your_passkey_here
MPESA_SHORTCODE=174379

# ngrok URL (leave as placeholder for now, we'll update it in Step 6)
MPESA_CALLBACK_URL=https://your-ngrok-url.ngrok-free.dev/mpesa/callback

# Debug Mode
DEBUG_MODE=true
```

**Save and exit** (Ctrl+X, then Y, then Enter for nano)

---

## Step 6: ngrok Setup

### 6.1 Install ngrok

**Mac/Linux:**
```bash
# Using brew
brew install ngrok

# Or download from https://ngrok.com/download
```

**Windows:** Download from https://ngrok.com/download and follow installation instructions

### 6.2 Authenticate ngrok

```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

Get your auth token from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)

### 6.3 Start ngrok Tunnel

```bash
ngrok http 8000
```

You should see output like:
```
Forwarding https://a1b2-c3d4-e5f6.ngrok-free.dev -> http://localhost:8000
```

**Copy the HTTPS URL** (e.g., `https://a1b2-c3d4-e5f6.ngrok-free.dev`) - you'll need it for the next step.

**Keep this terminal running!** Do not close it.

### 6.4 Update .env with ngrok URL

Open a new terminal, navigate to the project, and update `.env`:

```bash
nano .env
```

Replace `MPESA_CALLBACK_URL` with your actual ngrok URL:

```env
MPESA_CALLBACK_URL=https://a1b2-c3d4-e5f6.ngrok-free.dev/mpesa/callback
```

Replace `a1b2-c3d4-e5f6` with your actual ngrok subdomain.

---

## Step 7: Install Frontend Dependencies

```bash
cd fintech-app-ui
npm install
cd ..
```

---

## Step 8: Start the Backend Server

**Terminal 1** (make sure your virtual environment is active with `source venv/bin/activate`):

```bash
python -m src.run_api
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal running!**

---

## Step 9: Start the Frontend Server

**Terminal 2** (new terminal):

```bash
cd fintech-app-ui
npm run dev
```

You should see:
```
Ready in Xms
 ○ Local:   http://localhost:3000
```

**Keep this terminal running!**

---

## Step 10: Verify Everything is Working

### 10.1 Check Backend Health

Open your browser and visit:
```
http://localhost:8000/health
```

You should see: `{"status":"healthy"}`

### 10.2 Open the Application

Visit:
```
http://localhost:3000
```

You should see:
- ✅ Home page with camera feed
- ✅ Registration page accessible
- ✅ Sales dashboard showing analytics

### 10.3 Test Face Recognition

1. You'll need to register a user first (see below)
2. Navigate to Registration page
3. Allow camera access
4. Register a user (you'll be your first test user!)

---

## Quick User Registration (For Testing)

Since you're just starting, you need at least one registered user:

1. Navigate to `http://localhost:3000/enroll`
2. Click "Allow" when browser asks for camera permission
3. Enter your name
4. Position your face in the camera frame
5. Wait for 10 face samples to be captured
6. Click "Register"
7. You should see a success message

Now you can use this user for face recognition payments!

---

## Terminal Summary (What Should Be Running)

You should have **3 terminals** running simultaneously:

1. **Terminal 1:** ngrok tunnel (shows your public URL)
   ```bash
   ngrok http 8000
   ```

2. **Terminal 2:** Python backend (shows uvicorn is running)
   ```bash
   python -m src.run_api
   ```

3. **Terminal 3:** Next.js frontend (shows development server)
   ```bash
   cd fintech-app-ui && npm run dev
   ```

If you close any of these, that service will stop!

---

## Troubleshooting Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'vision'"

**Cause:** Virtual environment not activated or PYTHONPATH issue

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate

# Use the recommended startup method
python -m src.run_api
```

---

### Issue 2: NumPy/OpenCV Compatibility Errors

**Symptoms:**
- `numpy.core._exceptions._ArrayMemoryError`
- `cv2.imdecode` not working
- OpenCV can't read numpy arrays

**Cause:** NumPy 2.0+ installed (incompatible with OpenCV 4.x)

**Solution:**
```bash
# Downgrade NumPy
pip install "numpy<2.0"

# Verify the fix
python -c "import numpy; import cv2; import numpy; arr = numpy.zeros((100, 100, 3), dtype=numpy.uint8); cv2.imencode('.jpg', arr); print('Fixed!')"
```

---

### Issue 3: TensorFlow/Keras Import Errors

**Symptoms:**
- `ImportError: cannot import name 'keras' from 'tensorflow'`
- `AttributeError: module 'tensorflow' has no attribute 'keras'`

**Cause:** TensorFlow 2.16+ uses Keras 3.x, which requires tf-keras adapter

**Solution:**
```bash
pip install "tf-keras==2.16.0"
```

---

### Issue 4: Firebase "File not found" Error

**Cause:** `FIREBASE_CREDENTIALS` path in `.env` is incorrect

**Solution:**
1. Verify the file exists:
   ```bash
   ls -la /path/you/specified/in/.env
   ```

2. Use absolute path in `.env`:
   ```env
   FIREBASE_CREDENTIALS=/home/your-username/projects/Face-Pay/your-file-name.json
   ```

3. On Windows, use forward slashes:
   ```env
   FIREBASE_CREDENTIALS=C:/Users/your-username/projects/Face-Pay/your-file-name.json
   ```

---

### Issue 5: Camera Permission Denied

**Cause:** Browser blocking camera access

**Solution:**
1. Check browser address bar for camera icon (usually on the left)
2. Click the icon and select "Allow"
3. If not visible, go to browser settings → Site settings → Camera
4. Find `localhost:3000` and set to "Allow"
5. Refresh the page

---

### Issue 6: M-Pesa CallbackNot Working

**Cause:** ngrok URL incorrect or service not running

**Solution:**
1. Verify ngrok is running and showing HTTPS URL
2. Check `.env` has correct ngrok URL (include `/mpesa/callback` at end)
3. Test callback endpoint directly:
   ```bash
   curl -X POST https://your-ngrok-url.ngrok-free.dev/mpesa/callback -H "Content-Type: application/json" -d '{}'
   ```

---

### Issue 7: Server Port Already in Use

**Symptoms:** `OSError: [Errno 48] Address already in use`

**Cause:** Previous server instance still running

**Solution:**

Find and kill the process:
```bash
# Find process on port 8000
lsof -i :8000

# Kill the process (replace PID with actual ID)
kill -9 PID

# On Windows:
netstat -ano | findstr :8000
taskkill /PID PID /F
```

---

### Issue 8: npm install Hangs or Errors

**Solution:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Try again
npm install
```

---

## Quick Reference Commands

### Start Everything (3 terminals)

```bash
# Terminal 1: ngrok
ngrok http 8000

# Terminal 2: Backend
source venv/bin/activate
python -m src.run_api

# Terminal 3: Frontend
cd fintech-app-ui
npm run dev
```

### Stop Everything

Press `Ctrl+C` in each terminal (3 times)

### Restart Backend After Changes

```bash
# Stop the backend (Ctrl+C)
# Run again
python -m src.run_api
```

---

## Presentation Day Checklist

Before presenting, verify:

- [ ] All 3 terminals running (ngrok, backend, frontend)
- [ ] Backend health check passes: `http://localhost:8000/health`
- [ ] Frontend loads at: `http://localhost:3000`
- [ ] At least one test user registered in the system
- [ ] Camera permissions allowed in browser
- [ ] Notebook/computer connected to internet (for M-Pesa STK Push)
- [ ] ngrok HTTPS URL updated in `.env`
- [ ] Have a phone handy for M-Pesa STK Push testing

---

## Next Steps After Setup

Once everything is running:

1. **Register a test user** yourself through the Registration page
2. **Test face recognition** - try authenticating that user
3. **Test payment flow** - initiate a payment and complete it via M-Pesa
4. **Check dashboard** - verify transactions show up in Sales Analytics

---

## Need Help?

If you encounter issues not covered here:

1. Check the `docs/` folder for additional guides
2. Review the main `README.md` for more technical details
3. Check terminal error messages carefully - they usually indicate the exact problem

---

## Summary of Critical Version Requirements

| Package | Required Version | Why |
|---------|------------------|-----|
| Python | 3.11 - 3.12 | Compatible with all dependencies |
| NumPy | < 2.0 | NumPy 2.0+ breaks OpenCV 4.x |
| TensorFlow | cpu==2.17.1 | Stable for DeepFace & Python 3.12 |
| tf-keras | ==2.16.0 | Required adapter for TensorFlow 2.16+ |
| Node.js | 18+ | Required by Next.js 14+ |

**Never use `pip install --upgrade` without specifying versions** - this causes dependency conflicts!

---

Good luck with your presentation! 🚀