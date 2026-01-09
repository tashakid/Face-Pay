#!/usr/bin/env python
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database import db

db_user_id = 'natasha_wamuyu_'
result = db.delete_user(db_user_id)
print(f"Delete result for {db_user_id}: {result}")