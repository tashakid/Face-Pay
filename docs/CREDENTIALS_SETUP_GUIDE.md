# M-Pesa Sandbox Credentials Setup Guide

## Overview

This guide walks you through obtaining M-Pesa Daraja API sandbox credentials for the Face-Pay system.

## Prerequisites

- Safaricom M-Pesa active account (test phone number)
- Internet access
- Basic knowledge of APIs

## Step-by-Step Guide

### Step 1: Register for M-Pesa Daraja API

1. Visit the M-Pesa Daraja portal: https://developer.safaricom.co.ke/
2. Click on **"Register"** or **"Sign Up"**
3. Fill in your registration details:
   - **Name**: Your full name
   - **Email**: Your active email address
   - **Phone**: Your Safaricom phone number
   - **Password**: Create a strong password

4. Submit the registration form

### Step 2: Verify Your Account

1. Check your email for a verification link
2. Click the verification link to confirm your email
3. You may receive an SMS verification on your phone
4. Enter the verification code when prompted

### Step 3: Create an App

1. Log in to the Daraja portal
2. Navigate to **"My Apps"** or **"Apps"**
3. Click on **"Add New App"** or **"Create App"**
4. Fill in the app details:
   - **App Name**: Face-Pay (or any meaningful name)
   - **Description**: Face Recognition Payment System
   - **Environment**: Select "Sandbox"

5. Submit the app creation form

### Step 4: Obtain Sandbox Credentials

After creating the app:

1. Go to **"My Apps"** → **"Face-Pay"** (or your app name)
2. Look for the **"Credentials"** or **"Keys"** tab
3. You'll need these credentials:

   - **Consumer Key**: Your API key
   - **Consumer Secret**: Your API secret
   - **Passkey**: Available in your app details

### Step 5: Configure Shortcode

For testing in sandbox:

- Use the **default test shortcode**: `174379`
- This shortcode works automatically in sandbox mode

For production deployment:
- Contact Safaricom to obtain a live shortcode
- The shortcode will be a 4-6 digit number

### Step 6: Set Up Callback URL

Your callback URL is where M-Pesa will send payment results.

**For local testing:**

```
http://localhost:8000/mpesa/callback
```

**For production:**
```
https://your-domain.com/mpesa/callback
```

**Note:** For testing, you can use tools like:
- ngrok: https://ngrok.com/ (creates public URL from localhost)
- localtunnel: https://localtunnel.github.io/www/

### Step 7: Get Firebase API Key

1. Go to your Firebase project console: https://console.firebase.google.com/
2. Click on **Gear Icon** → **Project Settings**
3. Scroll down to **"Your apps"** section
4. Click on the **Web App** icon (`</>`)
5. Copy the **firebaseConfig** object
6. Extract the `apiKey` value

### Step 8: Download Firebase Service Account Key

1. Go to Firebase project settings
2. Click on **"Service Accounts"** tab
3. Click **"Generate new private key"**
4. Save the JSON file securely
5. This file contains: `project_id`, `private_key`, `client_email`

### Step 9: Configure Environment Variables

Create or update your `.env` file:

```env
# M-Pesa Darja API Configuration (Sandbox)
MPESA_KEY=your_consumer_key_here
MPESA_SECRET=your_consumer_secret_here
MPESA_PASSKEY=your_passkey_here
MPESA_SHORTCODE=174379
MPESA_CALLBACK_URL=http://localhost:8000/mpesa/callback

# Firebase Configuration
FIREBASE_CREDENTIALS=/path/to/your/firebase_credentials.json
FIREBASE_API_KEY=your_firebase_api_key_here

# JWT Configuration
JWT_SECRET=your-jwt-secret-key-change-in-production

# Camera Configuration
CAMERA_INDEX=0
```

## Testing Your Credentials

### Test 1: Get OAuth Token

```bash
curl -X GET "https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials" \
  -H "Authorization: Basic $(echo -n "YOUR_KEY:YOUR_SECRET" | base64)"
```

Expected response:
```json
{
  "access_token": "your_access_token",
  "expires_in": "3599"
}
```

### Test 2: Send Test STK Push

```python
import requests

# Replace with your access token
access_token = "YOUR_ACCESS_TOKEN"

# STK Push URL
url = "https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

payload = {
    "BusinessShortCode": "174379",
    "Password": "YOUR_GENERATED_PASSWORD",
    "Timestamp": "20231230120000",
    "TransactionType": "CustomerPayBillOnline",
    "Amount": 1,
    "PartyA": "2547XXXXXXXX",
    "PartyB": "174379",
    "PhoneNumber": "2547XXXXXXXX",
    "CallBackURL": "http://your-callback-url.com/mpesa/callback",
    "AccountReference": "FacePay",
    "TransactionDesc": "Test Payment"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

## Common Issues and Solutions

### Issue 1: "Invalid Consumer Key" Error

**Solution:**
- Verify your Consumer Key is correct
- Ensure you're using sandbox credentials, not production
- Check that your app is enabled for sandbox

### Issue 2: "CORS Error" on Callback

**Solution:**
- Use HTTPS callbacks (required for production)
- Ensure your callback URL is publicly accessible
- For local testing, use ngrok or similar

### Issue 3: "InitiatorAccountNotFound" Error

**Solution:**
- This usually means incorrect passkey
- Verify MPESA_PASSKEY in your .env file
- Check that MPESA_SHORTCODE is correct: `174379`

### Issue 4: Callback Not Receiving Data

**Solution:**
- Ensure callback URL is publicly accessible
- Check firewall settings
- Verify callback endpoint is running
- Check server logs for errors

### Issue 5: Firebase Initialization Error

**Solution:**
- Verify `FIREBASE_CREDENTIALS` path is correct
- Ensure the JSON file has correct permissions (readable)
- Check that the service account has proper Firebase roles

## Security Best Practices

1. **Never commit credentials to version control**
   - Add `.env` to `.gitignore`
   - Rotate credentials if accidentally exposed

2. **Use environment variable files**
   - Keep .env files local
   - Use secrets management in production (AWS Secrets Manager, Azure Key Vault)

3. **Rotate credentials regularly**
   - Change API keys periodically
   - Update credentials in deployment safely

4. **Limit permissions**
   - Use least-privilege access for Firebase service accounts
   - Only grant necessary API permissions

5. **Monitor usage**
   - Set up usage alerts
   - Review API logs regularly
   - Detect unusual activity

## Production Deployment Checklist

- [ ] Obtain production M-Pesa credentials from Safaricom
- [ ] Use production shortcode (not 174379)
- [ ] Set up HTTPS with SSL certificate
- [ ] Use production Firebase project
- [ ] Implement proper error handling
- [ ] Add logging and monitoring
- [ ] Set up database backups
- [ ] Configure security headers (CORS, CSP)
- [ ] Test callback endpoint with real payments
- [ ] Set up rate limiting
- [ ] Implement retry logic for failed transactions
- [ ] Enable webhook verification signatures

## Additional Resources

- M-Pesa Daraja Documentation: https://developer.safaricom.co.ke/APIs
- M-Pesa API Support: https://developer.safaricom.co.ke/support
- Firebase Documentation: https://firebase.google.com/docs/
- ngrok Documentation: https://ngrok.com/docs

## Support and Troubleshooting

If you encounter issues:

1. Check the error message carefully
2. Verify all credentials in .env file
3. Review API logs for detailed error information
4. Test your callback URL using online webhook testers
5. Contact M-Pesa Daraja support for account-level issues
6. Check Firebase console for project configuration issues

## Summary

This guide helps you:
- ✅ Register for M-Pesa Daraja API
- ✅ Obtain sandbox credentials
- ✅ Configure Firebase for authentication
- ✅ Set up environment variables
- ✅ Test the integration
- ✅ Troubleshoot common issues
- ✅ Prepare for production deployment

Once configured, your Face-Pay system can:
- Process M-Pesa payments via STK Push
- Receive payment callbacks and update transaction status
- Authenticate users via Firebase
- Store user data and payment history in Firestore
- Enable face recognition for secure payments

---

**Last Updated**: December 2025
**Version**: 1.0