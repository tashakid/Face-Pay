import os
print(f"DEBUG_MODE env var before: {os.getenv('DEBUG_MODE')}")
os.environ['DEBUG_MODE'] = 'true'
print(f"DEBUG_MODE env var after: {os.getenv('DEBUG_MODE')}")

import sys
sys.path.insert(0, '/home/desk-fam/projects/Face-Pay/src')
from debug_mode import configure_debug_mode
enabled = configure_debug_mode()
print(f"Debug mode enabled: {enabled}")
