# M-Pesa Daraja Sandbox Setup Guide

This guide will help you get your M-Pesa sandbox credentials for testing the Face Recognition Payment System.

## 📋 Prerequisites

- A valid phone number (Safaricom line)
- Email address for account registration
- Internet connection

## 🔐 Step-by-Step Guide

### Step 1: Access the Safaricom Developer Portal

1. Go to [https://developer.safaricom.co.ke/](https://developer.safaricom.co.ke/)
2. Click on **"Login"** or **"Sign Up"** if you don't have an account
3. Fill in your details:
   - **Email**: Your email address
   - **Password**: Create a strong password
   - **Full Name**: Your full name
   - **Phone Number**: Your Safaricom phone number (format: 2547XXXXXXXX)
4. Click **"Sign Up"**

### Step 2: Verify Your Account

1. Check your email for a verification link
2. Click the verification link to activate your account
3. Log in with your credentials

### Step 3: Create a New App

1. After logging in, click on **"My Apps"** in the navigation menu
2. Click on **"Add New App"** button
3. Fill in the app details:
   - **App Name**: `Face Payment System` (or any name you prefer)
   - **Description**: `Face Recognition Payment System for capstone project`
   - **Organization**: Your organization or project name
4. Click **"Submit"**

### Step 4: Get Consumer Key and Secret

1. After creating the app, you'll see your app details
2. Copy your **Consumer Key** (this is your `MPESA_KEY`)
3. Copy your **Consumer Secret** (this is your `MPESA_SECRET`)

**Important:** Keep these credentials secure and never share them publicly!

### Step 5: Get Lipa Na M-Pesa Passkey

1. Click on **"My Apps"** in the navigation menu
2. Find your app and click on it
3. Scroll down to **"Test Credentials"** section
4. Expand **"Lipa Na M-Pesa Online"**
5. Copy your **Passkey** (this is your `MPESA_PASSKEY`)

### Step 6: Configure Your `.env` File

Open the `.env` file in your project and update it with your credentials:

```env
# M-Pesa API Configuration (Safaricom Daraja - Sandbox)
MPESA_KEY=your_consumer_key_here
MPESA_SECRET=your_consumer_secret_here
MPESA_PASSKEY=your_passkey_here
MPESA_SHORTCODE=174379
MPESA_CALLBACK_URL=https://sandbox.safaricom.co.ke/mpesa/
```

Replace the placeholder values with your actual credentials:
- `your_consumer_key_here` → Your Consumer Key from Step 4
- `your_consumer_secret_here` → Your Consumer Secret from Step 4
- `your_passkey_here` → Your Passkey from Step 5

## 📱 Step 7: Test Your Phone Number

Before running the application, ensure your phone number is registered for M-Pesa:

1. Dial `*234#` on your Safaricom line
2. Select **"Lipa Na M-Pesa"**
3. Select **"Activate M-Pesa"**
4. Enter your M-Pesa PIN
5. You should see a success message

## 🧪 Step 8: Test the Application

Run the application:

```bash
python src/main.py
```

Choose option **1** (Full Camera Workflow) or **2** (Demo Mode).

When the payment is triggered, you should receive an STK Push prompt on your phone:

```
💳 Initiating Payment via M-Pesa STK Push...
✅ STK Push Sent! Please check your phone to enter PIN.
   Amount: KES 1000.00
   Phone: 254712345678
   MerchantRequestID: ws_CO_...
   CheckoutRequestID: ws_CO_...
```

## 📊 Sandbox vs Production

| Feature | Sandbox | Production |
|---------|---------|------------|
| URL | `sandbox.safaricom.co.ke` | `api.safaricom.co.ke` |
| Shortcode | `174379` | Your production shortcode |
| Callback | `sandbox.safaricom.co.ke/mpesa/` | Your production callback URL |
| Real Money | No (test mode) | Yes |
| Phone Number | Any Safaricom number | Any Safaricom number |

## 🔍 Troubleshooting

### Issue: "Invalid Credentials" Error

**Solution:**
- Verify your Consumer Key and Secret are correct
- Check for extra spaces in the credentials
- Ensure the app is active in the Daraja portal

### Issue: "Network Error" or Connection Timeout

**Solution:**
- Check your internet connection
- Verify the sandbox URL is correct
- Try again after a few minutes

### Issue: "No STK Push Received on Phone"

**Solution:**
- Ensure your phone number is in format `2547XXXXXXXX`
- Verify M-Pesa is activated on your line
- Check if you have sufficient balance (for testing, use KES 1-10)
- Ensure your phone has network signal

### Issue: "Callback URL Error"

**Solution:**
- For sandbox testing, use the default callback URL
- The callback URL is not critical for basic STK Push testing
- The system will work without a live callback server

## 📞 Contact Support

If you encounter issues:

1. **Safaricom Developer Support**: [developer@safaricom.co.ke](mailto:developer@safaricom.co.ke)
2. **Daraja Documentation**: [https://developer.safaricom.co.ke/APIs](https://developer.safaricom.co.ke/APIs)
3. **Daraja Community**: [https://developer.safaricom.co.ke/Community](https://developer.safaricom.co.ke/Community)

## ✅ Verification Checklist

Before testing, ensure:

- [ ] You have a Safaricom developer account
- [ ] You have created an app in the Daraja portal
- [ ] You have copied Consumer Key and Secret
- [ ] You have copied the Lipa Na M-Pesa Passkey
- [ ] You have updated the `.env` file with your credentials
- [ ] Your phone number is activated for M-Pesa
- [ ] Your phone has network signal
- [ ] You have sufficient balance (KES 10+ for testing)

## 🎯 Next Steps

Once you have your credentials configured:

1. Run `python src/main.py`
2. Choose option **1** (Full Camera Workflow)
3. Complete face recognition
4. Wait for STK Push on your phone
5. Enter your M-Pesa PIN when prompted
6. Complete gesture confirmation
7. Transaction completed!

---

**Note:** The sandbox environment is for testing only. No real money will be deducted. For production use, you'll need to:
- Get a production shortcode from Safaricom
- Switch to production API URLs
- Implement a live callback server
- Get approval from Safaricom for production use