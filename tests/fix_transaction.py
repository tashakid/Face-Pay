#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/desk-fam/projects/Face-Pay/src')
from database import db

checkout_request_id = "ws_CO_08012026235302989115313649"
update_data = {
    "status": "completed",
    "receipt_number": "UA88Z37JET",
    "completed_amount": 1,
    "customer_phone": 254115313649,
    "transaction_date": "20260108235329"
}
result = db.update_transaction(checkout_request_id, update_data)
print(f"Result: {result}")
