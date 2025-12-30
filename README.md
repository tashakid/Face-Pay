# Face Recognition Payment System

A cutting-edge contactless payment solution that leverages facial recognition technology combined with hand gesture confirmation for secure and seamless transactions.

## Project Purpose

This system facilitates contactless payments using facial recognition and hand gesture confirmation. Users can authenticate payments through facial recognition and confirm or cancel transactions using simple hand gestures:
- **Open Hand** = Confirm Payment
- **Fist** = Cancel Payment

The system provides a hygienic, fast, and secure payment experience ideal for retail environments, restaurants, and service industries where contactless transactions are preferred.

## Infrastructure

### Hardware
- **Camera**: High-definition camera for facial recognition and gesture detection
- **POS Terminal**: Point of Sale terminal for transaction processing

### Software
- **OpenCV (Vision)**: Computer vision library for image processing and facial detection
- **SFace (Model)**: Deep learning model for robust facial recognition
- **MediaPipe (Gestures)**: Hand tracking and gesture recognition framework

### Backend
- **M-Pesa API (Payments)**: Integration with M-Pesa mobile payment system for transaction processing
- **Firebase (Database)**: Cloud database for user authentication, transaction records, and system data

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- Firebase project credentials
- M-Pesa API credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/face-payment-backend.git
   cd face-payment-backend
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your actual credentials:
   - `MPESA_KEY`: Your M-Pesa API key
   - `MPESA_SECRET`: Your M-Pesa API secret
   - `FIREBASE_CREDENTIALS`: Path to your Firebase service account JSON file

5. **Run the application**
   ```bash
   cd src
   python main.py
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info

### Vision
- `POST /api/vision/detect-face` - Detect faces in image
- `POST /api/vision/register-face` - Register user's face
- `POST /api/vision/recognize-face` - Recognize face from image

### Payment
- `POST /api/payment/process` - Process payment
- `GET /api/payment/status/{transaction_id}` - Get transaction status
- `POST /api/payment/callback` - M-Pesa callback handler

## Collaborators (Group 5)

- **Natasha Wamuyu**
- **Faith Sang**
- **John Mwangi**
- **Victor Muchina**
- **Victor Ngunyi**
- **Paul Owiti**

## License

This project is part of a capstone project for educational purposes.

## Support

For technical support or inquiries, please contact the development team.