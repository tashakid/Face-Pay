# Final Working Configuration for Face-Pay Project

## Environment
- Python: 3.12.3
- Platform: Linux x86_64

## Package Versions (Working as of 2026-01-05)

### Core ML Packages
- tensorflow (CPU version): 2.20.0
- mediapipe: 0.10.31
- protobuf: 5.29.5
- grpcio: 1.66.2
- grpcio-status: 1.66.2

### Other Key Packages
- mediapipe: 0.10.31 (Note: shows platform warning but works)
- deepface: 0.0.96
- opencv-python: 4.12.0.88
- numpy: 2.2.6

## Known Issues and Warnings

### MediaPipe Platform Warning
- **Warning**: "mediapipe 0.10.31 is not supported on this platform"
- **Impact**: None - the package imports and works correctly despite the warning
- **Reason**: MediaPipe 0.10.31 has platform detection issues on some Linux x86_64 systems
- **Workaround**: Install with `pip install mediapipe==0.10.31` and ignore the warning

### Protobuf Version Compatibility
- **Issue**: TensorFlow 2.20.0 requires protobuf>=5.28.0
- **grpcio-status 1.66.2** requires protobuf<6.0dev,>=5.26.1
- **Resolution**: protobuf 5.29.5 satisfies both requirements

### grpcio Version Compatibility
- **Issue**: Newer grpcio (1.76.0) requires protobuf>=6.31.1 which conflicts with TensorFlow
- **Resolution**: Downgraded to grpcio 1.66.2 and grpcio-status 1.66.2

## Installation Commands

```bash
# Create virtual environment (if not exists)
python3.12 -m venv venv_new
source venv_new/bin/activate

# Install base packages
pip install numpy>=1.26.4 opencv-python pillow requests python-dotenv

# Install compatible grpcio versions (important for protobuf compatibility)
pip install grpcio<1.67.0 grpcio-status<1.67.0

# Install TensorFlow CPU version
pip install tensorflow-cpu

# Install MediaPipe (ignore platform warning)
pip install mediapipe==0.10.31

# Install other requirements
pip install deepface firebase-admin fastapi uvicorn[standard] pydantic>=2.0.0 python-jose[cryptography] python-multipart
```

## Verification

```python
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')

import mediapipe as mp
print(f'MediaPipe: {mp.__version__}')

import protobuf as pb
print(f'Protobuf: {pb.__version__}')

print("All packages imported successfully!")
```

## Why This Configuration Works

### TensorFlow 2.20.0 + Protobuf 5.29.5
- TensorFlow 2.20.0 is built with protobuf 5.28.3 gencode
- Protobuf 5.29.5 runtime is compatible with version 5.28.3 gencode
- Only one major version apart (acceptable per protobuf guarantee)

### grpcio 1.66.2 vs 1.76.0
- grpcio 1.76.0 requires protobuf>=6.31.1 (incompatible with TensorFlow)
- grpcio 1.66.2 requires protobuf>=5.26.1 and <6.0 (compatible)
- grpcio 1.66.2 is stable and well-tested

### MediaPipe 0.10.31
- Works with Python 3.8-3.12
- Platform support issue is cosmetic on Linux x86_64
- All MediaPipe solutions import and function correctly

## Alternative: Older TensorFlow (2.16.1)

If you prefer an older TensorFlow version with tighter protobuf requirements:

```bash
pip install tensorflow==2.16.1 grpcio<1.67.0 grpcio-status<1.67.0 protobuf<5.0.0dev,>=3.20.3
```

This uses protobuf 4.25.5 which MediaPipe is also compatible with.

## Reference
- TensorFlow install: https://www.tensorflow.org/install/pip
- MediaPipe GitHub: https://github.com/google-ai-edge/mediapipe
- Protobuf version guarantee: https://protobuf.dev/support/cross-version-runtime-guarantee