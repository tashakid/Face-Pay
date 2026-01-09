"""
Real M-Pesa STK Push Payment Processing Module for Face Recognition Payment System
Integrates with Safaricom Daraja API for actual payment processing
"""

import requests
import base64
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from dotenv import load_dotenv
import os

load_dotenv()

from database import db

app = FastAPI()

# M-Pesa Daraja API Configuration (Sandbox)
MPESA_KEY = os.getenv("MPESA_KEY", "")
MPESA_SECRET = os.getenv("MPESA_SECRET", "")
MPESA_PASSKEY = os.getenv("MPESA_PASSKEY", "")
MPESA_SHORTCODE = os.getenv("MPESA_SHORTCODE", "174379")
MPESA_CALLBACK_URL = os.getenv("MPESA_CALLBACK_URL", "http://localhost:8000/mpesa/callback")

# API Endpoints
OAUTH_URL = "https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials"
STK_PUSH_URL = "https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest"

# Token cache
access_token_cache = {"token": None, "expires_at": None}


class PaymentRequest(BaseModel):
    amount: float
    phone_number: str
    user_id: str
    description: Optional[str] = "Face Recognition Payment"


class PaymentResponse(BaseModel):
    transaction_id: str
    status: str
    message: str
    receipt: str
    timestamp: datetime


def get_access_token() -> str:
    """
    Get OAuth access token from Safaricom Daraja API.
    
    Returns:
        str: Access token for API authentication
    """
    global access_token_cache
    
    # Check if we have a valid cached token
    if access_token_cache["token"] and access_token_cache["expires_at"]:
        try:
            if datetime.now().timestamp() < access_token_cache["expires_at"]:
                return access_token_cache["token"]
        except TypeError:
            # Clear cache if there's a type mismatch (e.g., datetime vs float)
            print("⚠️  Token cache type mismatch, clearing cache...")
            access_token_cache["token"] = None
            access_token_cache["expires_at"] = None
    
    try:
        # Make OAuth request
        auth = (MPESA_KEY, MPESA_SECRET)
        headers = {"Content-Type": "application/json"}
        
        response = requests.get(OAUTH_URL, auth=auth, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        access_token = data.get("access_token")
        expires_in = int(data.get("expires_in", 3599))  # Default 1 hour, convert to int
        
        # Cache the token
        access_token_cache["token"] = access_token
        access_token_cache["expires_at"] = datetime.now().timestamp() + expires_in
        
        return access_token
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get access token: {str(e)}"
        )


def generate_password() -> str:
    """
    Generate base64 encoded password for STK Push.
    Password = Base64(Shortcode + Passkey + Timestamp)
    
    Returns:
        str: Base64 encoded password
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    data = f"{MPESA_SHORTCODE}{MPESA_PASSKEY}{timestamp}"
    password = base64.b64encode(data.encode()).decode()
    return password


def format_phone_number(phone: str) -> str:
    """
    Format phone number to M-Pesa format (2547XXXXXXXX).
    
    Args:
        phone: Phone number string
        
    Returns:
        str: Formatted phone number
    """
    # Remove any non-digit characters
    phone = ''.join(filter(str.isdigit, phone))
    
    # Handle different formats
    if phone.startswith('0'):
        phone = '254' + phone[1:]
    elif phone.startswith('+'):
        phone = phone[1:]
    elif not phone.startswith('254'):
        phone = '254' + phone
    
    return phone


class MpesaPaymentGateway:
    """
    Real M-Pesa STK Push Payment Gateway using Safaricom Daraja API
    """
    
    def __init__(self):
        self.transaction_count = 0
    
    def trigger_stk_push(self, phone_number: str, amount: float) -> Dict:
        """
        Trigger M-Pesa STK Push payment request.
        
        Args:
            phone_number: Phone number to send payment request to
            amount: Amount to charge
            
        Returns:
            Dictionary containing payment result
        """
        try:
            # Format phone number
            formatted_phone = format_phone_number(phone_number)
            
            # Get access token
            access_token = get_access_token()
            
            # Generate password
            password = generate_password()
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            
            # Prepare STK Push request payload
            payload = {
                "BusinessShortCode": MPESA_SHORTCODE,
                "Password": password,
                "Timestamp": timestamp,
                "TransactionType": "CustomerPayBillOnline",
                "Amount": int(amount),
                "PartyA": formatted_phone,
                "PartyB": MPESA_SHORTCODE,
                "PhoneNumber": formatted_phone,
                "CallBackURL": MPESA_CALLBACK_URL,
                "AccountReference": "FacePayment",
                "TransactionDesc": "Face Recognition Payment"
            }
            
            # Make STK Push request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(STK_PUSH_URL, json=payload, headers=headers)
            
            # Log the response for debugging
            print(f"📡 M-Pesa API Response Status: {response.status_code}")
            print(f"📡 M-Pesa API Response Body: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check response code
            response_code = data.get("ResponseCode")
            response_desc = data.get("ResponseDescription", "")
            merchant_request_id = data.get("MerchantRequestID", "")
            checkout_request_id = data.get("CheckoutRequestID", "")
            customer_message = data.get("CustomerMessage", "")
            
            if response_code == "0":
                # Success
                self.transaction_count += 1
                
                result = {
                    "success": True,
                    "message": "Request sent to phone",
                    "merchant_request_id": merchant_request_id,
                    "checkout_request_id": checkout_request_id,
                    "customer_message": customer_message,
                    "response_description": response_desc,
                    "phone_number": formatted_phone,
                    "amount": amount,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"✅ STK Push Sent! Please check your phone to enter PIN.")
                print(f"   Amount: KES {amount:.2f}")
                print(f"   Phone: {formatted_phone}")
                print(f"   MerchantRequestID: {merchant_request_id}")
                print(f"   CheckoutRequestID: {checkout_request_id}")
                
                return result
            else:
                # Failure
                error_result = {
                    "success": False,
                    "error": response_desc,
                    "response_code": response_code,
                    "merchant_request_id": merchant_request_id,
                    "checkout_request_id": checkout_request_id,
                    "customer_message": customer_message
                }
                
                print(f"❌ STK Push Failed: {response_desc}")
                print(f"   Response Code: {response_code}")
                
                return error_result
                
        except requests.exceptions.RequestException as e:
            error_result = {
                "success": False,
                "error": f"Network error: {str(e)}",
                "receipt": None
            }
            print(f"❌ Network Error: {str(e)}")
            return error_result
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Payment failed: {str(e)}",
                "receipt": None
            }
            print(f"❌ Payment Error: {str(e)}")
            return error_result
    
    def get_transaction_status(self, transaction_id: str) -> Dict:
        """
        Get transaction status (placeholder for future implementation).
        
        Args:
            transaction_id: Transaction ID to check
            
        Returns:
            Transaction status information
        """
        # This would require implementing the STK Push Query API
        # For now, return a mock response
        return {
            "transaction_id": transaction_id,
            "status": "pending",
            "message": "Transaction status check not yet implemented",
            "timestamp": datetime.now().isoformat()
        }


class PaymentService:
    """
    Service layer for payment operations using M-Pesa STK Push
    """
    
    def __init__(self):
        self.payment_gateway = MpesaPaymentGateway()
    
    def process_payment(self, payment_request: PaymentRequest) -> PaymentResponse:
        """
        Process a payment request using M-Pesa STK Push.
        
        Args:
            payment_request: Payment request details
            
        Returns:
            Payment response with transaction details
        """
        try:
            # Process payment through M-Pesa gateway
            result = self.payment_gateway.trigger_stk_push(
                payment_request.phone_number,
                payment_request.amount
            )

            if result["success"]:
                transaction_id = result.get("checkout_request_id", result.get("merchant_request_id"))

                transaction_data = {
                    "checkout_request_id": transaction_id,
                    "merchant_request_id": result.get("merchant_request_id"),
                    "user_id": payment_request.user_id,
                    "amount": payment_request.amount,
                    "phone_number": payment_request.phone_number,
                    "status": "pending",
                    "message": result["message"],
                    "description": payment_request.description,
                    "timestamp": datetime.now().isoformat(),
                    "created_at": datetime.now()
                }

                db.create_transaction(transaction_data)

                return PaymentResponse(
                    transaction_id=transaction_id,
                    status="pending",
                    message=result["message"],
                    receipt=transaction_id,
                    timestamp=datetime.now()
                )
            else:
                raise HTTPException(status_code=400, detail=result.get("error", "Payment failed"))
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Payment processing failed: {str(e)}")
    
    def get_transaction_status(self, transaction_id: str) -> Dict:
        transaction = db.get_transaction(transaction_id)
        if transaction:
            return transaction
        else:
            raise HTTPException(status_code=404, detail="Transaction not found")

    def get_user_transactions(self, user_id: str) -> List[Dict]:
        return db.get_user_transactions(user_id)
    
    def get_all_transactions(self) -> List[Dict]:
        return db.get_all_transactions()
    
    def handle_callback(self, callback_data: Dict) -> Dict:
        if 'Body' not in callback_data or 'stkCallback' not in callback_data['Body']:
            return {"status": "error", "message": "Invalid callback format"}

        stk_callback = callback_data['Body']['stkCallback']
        merchant_request_id = stk_callback.get('MerchantRequestID')
        checkout_request_id = stk_callback.get('CheckoutRequestID')
        result_code = stk_callback.get('ResultCode')
        result_desc = stk_callback.get('ResultDesc', '')

        status = "completed" if str(result_code) == "0" else "failed"

        update_data = {
            "status": status,
            "result_code": result_code,
            "result_description": result_desc,
            "updated_at": datetime.now()
        }

        if status == "completed" and 'CallbackMetadata' in stk_callback:
            metadata = stk_callback['CallbackMetadata']['Item']
            mpesa_receipt = None
            mpesa_amount = None
            mpesa_phone = None
            transaction_date = None

            for item in metadata:
                if item['Name'] == 'MpesaReceiptNumber':
                    mpesa_receipt = item['Value']
                elif item['Name'] == 'Amount':
                    mpesa_amount = item['Value']
                elif item['Name'] == 'PhoneNumber':
                    mpesa_phone = item['Value']
                elif item['Name'] == 'TransactionDate':
                    transaction_date = item['Value']

            update_data.update({
                "receipt_number": mpesa_receipt,
                "completed_amount": mpesa_amount,
                "customer_phone": mpesa_phone,
                "transaction_date": transaction_date
            })

        db.update_transaction(checkout_request_id or merchant_request_id, update_data)

        return {
            "status": "success",
            "message": "Callback processed",
            "transaction_id": checkout_request_id,
            "result_code": result_code,
            "result_desc": result_desc
        }


# Initialize payment service
payment_service = PaymentService()

# API Endpoints
@app.post("/process", response_model=PaymentResponse)
async def process_payment(payment_request: PaymentRequest):
    """Process a payment request"""
    return payment_service.process_payment(payment_request)

@app.get("/status/{transaction_id}")
async def get_payment_status(transaction_id: str):
    """Get payment transaction status"""
    return payment_service.get_transaction_status(transaction_id)

@app.post("/callback")
async def payment_callback(callback_data: dict):
    """Handle M-Pesa callback"""
    return payment_service.handle_callback(callback_data)

@app.get("/transactions/{user_id}")
async def get_user_transactions(user_id: str):
    """Get all transactions for a user"""
    return {"transactions": payment_service.get_user_transactions(user_id)}

@app.get("/health")
async def payment_health():
    return {
        "status": "healthy",
        "service": "M-Pesa STK Push Payment Gateway",
        "shortcode": MPESA_SHORTCODE
    }

router = app