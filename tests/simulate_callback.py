#!/usr/bin/env python3
"""Manual callback simulator - use this to test transaction updates without ngrok"""

import requests
import sys

def simulate_callback(checkout_request_id: str, result_code: int = "0"):
    """Simulate M-Pesa callback for testing"""

    callback_data = {
        "Body": {
            "stkCallback": {
                "MerchantRequestID": "5509-4da0-9b55-10960d52c66914503",
                "CheckoutRequestID": checkout_request_id,
                "ResultCode": result_code,
                "ResultDesc": "The service request is processed successfully." if result_code == "0" else "Request cancelled by user"
            }
        }
    }

    # Add metadata for successful transactions
    if result_code == "0":
        callback_data["Body"]["stkCallback"]["CallbackMetadata"] = {
            "Item": [
                {"Name": "Amount", "Value": 1},
                {"Name": "MpesaReceiptNumber", "Value": "LIP6X8Y7Y3"},
                {"Name": "Balance", "Value": "0"},
                {"Name": "TransactionDate", "Value": "20260108232016"},
                {"Name": "PhoneNumber", "Value": 254115313649}
            ]
        }

    print(f"Sending callback for: {checkout_request_id}")
    print(f"Result Code: {result_code}")
    print("-" * 60)

    try:
        response = requests.post(
            "http://localhost:8000/mpesa/callback",
            json=callback_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Callback Status: {response.status_code}")
        print(f"Callback Response: {response.json()}")
        print("\n✅ Callback sent successfully!")
    except Exception as e:
        print(f"❌ Error sending callback: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulate_callback.py <checkout_request_id> [result_code]")
        print("Example (successful): python simulate_callback.py ws_CO_08012026232034575115313649 0")
        print("Example (failed): python simulate_callback.py ws_CO_08012026232034575115313649 2001")
        sys.exit(1)

    checkout_request_id = sys.argv[1]
    result_code = sys.argv[2] if len(sys.argv) > 2 else "0"

    simulate_callback(checkout_request_id, result_code)