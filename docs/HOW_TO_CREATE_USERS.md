# How to Create Users and Test Face-Pay

## Step 1: Set Up IP Camera on Your Phone

### On your phone (with IP Webcam app):
1. Make sure your phone and computer are on the **same WiFi network**
2. Open the IP Webcam app
3. Scroll down and tap **"Start Server"**
4. Note the IP address shown (e.g., `http://192.168.1.5:8080`)

### Test the camera:
1. Open this URL in your computer's browser:
   ```
   http://YOUR_PHONE_IP:8080/video
   ```
   Replace `YOUR_PHONE_IP` with the actual IP from your phone
2. You should see your phone's camera feed

### Configure in the app:
```bash
# Edit the .env file
nano .env
```

Update the last line:
```env
CAMERA_URL=http://192.168.X.X:8080/video
```
Replace with your actual phone IP.

**Important**: Make sure the IP Webcam app stays running while using Face-Pay.

---

## Step 2: Start the Face-Pay Server

```bash
cd /home/desk-fam/projects/Face-Pay
source venv/bin/activate
python src/main.py
```

Select option **3** (API Server mode)

The server will start at `http://localhost:8000`

---

## Step 3: Create Users

### Method A: Using the Web Form (Easiest)

1. Open this file in your browser:
   ```
   file:///home/desk-fam/projects/Face-Pay/create_registration_client.html
   ```

2. Fill in the form:
   - **Name**: Full name (e.g., "John Doe")
   - **Email**: Unique email (e.g., "john@example.com")
   - **Phone**: M-Pesa phone number (format: `2547XXXXXXXX`)
   - **Password**: Set a password
   - **Face Photo**: Take or upload a clear face photo

3. Click **Register User**

4. Save the **User ID** shown (you'll need it later)

5. Repeat for more users with different phones/emails

### Method B: Using Swagger UI

1. Open `http://localhost:8000/docs` in your browser
2. Scroll to **Registration** section
3. Expand `POST /registration/register-with-face`
4. Click **Try it out**
5. Fill in the request body:
   ```json
   {
     "name": "John Doe",
     "email": "john@example.com",
     "phone_number": "254712345678",
     "password": "password123"
   }
   ```
6. Upload a face photo using the file input
7. Click **Execute**

---

## Step 4: Test Face Recognition

### Using Swagger UI:

1. Open `http://localhost:8000/docs`
2. Expand `POST /registration/verify-face`
3. Try it out
4. Upload the same face photo you used for registration
5. Click **Execute**

You should get a response like:
```json
{
  "success": true,
  "user_id": "abc123...",
  "confidence": 0.95
}
```

---

## Step 5: Test Payment with Face Recognition

### Start the Camera Mode (with IP Camera):

```bash
# In a new terminal, keep the API server running in another terminal
cd /home/desk-fam/projects/Face-Pay
source venv/bin/activate
python src/main.py
```

Select option **1** (Complete Face Recognition Payment Workflow)

The app will:
1. Connect to your phone's camera
2. Detect your face from live video
3. Match it against registered users in Firebase
4. Send M-Pesa STK Push to your phone
5. Ask you to complete payment on your phone

### Test Multiple Users:

1. Register 3-5 different users using the web form
2. Each user should have:
   - Different face photo
   - Different phone number
   - Different email

3. Test by having each person:
   - Stand in front of the IP camera
   - Let the app recognize them
   - Receive the STK Push on their phone

---

## API Reference for Testing

### Register New User
```
POST /registration/register-with-face
```

### Add Face to Existing User
```
POST /registration/register-face-only
{
  "user_id": "existing_user_id",
  "image": [photo file]
}
```

### Verify Face
```
POST /registration/verify-face
{
  "image": [photo file]
}
```

### Get User Info
```
GET /user/{user_id}
```

### Initiate Payment (after face recognition)
```
POST /mpesa/process
{
  "phone_number": "2547XXXXXXXX",
  "amount": 100.0
}
```

---

## Troubleshooting

### Camera Issues:
- **"Failed to connect to IP camera"**:
  - Check if phone WiFi is ON
  - Verify phone and computer on same network
  - Make sure IP Webcam app is running
  - Try the URL in browser first

### Face Recognition Issues:
- **"No matching face found"**:
  - Make sure you used the same face photo for registration
  - Use clear, well-lit photos
  - Avoid wearing accessories (glasses, masks) between registration and recognition

### Payment Issues:
- **STK Push not received**:
  - Verify phone number is in format: `2547XXXXXXXX` (not +254 or 07...)
  - Check if M-Pesa credentials are correct
  - Ensure the CALLBACK_URL is accessible
- **Payment fails**:
  - Make sure you have test funds on M-Pesa sandbox
  - Enter the correct PIN on your phone

---

## Production Readiness Checklist

- [ ] Multiple users can register successfully
- [ ] Face recognition matches the correct users
- [ ] M-Pesa STK Push is received on correct phone numbers
- [ ] Payments complete successfully
- [ ] Callback URL receives transaction status
- [ ] Transactions are saved to Firebase
- [ ] Gesture confirmation works (camera mode)

---

## Quick Test Script

After registering users, test them all:

```bash
# Test user 1
curl -X POST "http://localhost:8000/mpesa/process" \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "254712345678",
    "amount": 100.0
  }'

# Test user 2
curl -X POST "http://localhost:8000/mpesa/process" \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "254723456789",
    "amount": 200.0
  }'
```