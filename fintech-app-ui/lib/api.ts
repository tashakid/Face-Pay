const BASE_URL = 'http://localhost:8000';

export interface FaceRecognitionRequest {
  face_image: string;
}

export interface FaceRecognitionResponse {
  success: boolean;
  user_id?: string;
  name?: string;
  phone_number?: string;
  email?: string;
  confidence?: number;
  message?: string;
}

export interface PaymentRequest {
  amount: number;
  phone_number: string;
  user_id: string;
  description?: string;
}

export interface PaymentResponse {
  transaction_id: string;
  status: string;
  message: string;
  receipt: string;
  timestamp: string;
}

export interface Transaction {
  checkout_request_id?: string;
  merchant_request_id?: string;
  user_id: string;
  amount: number;
  phone_number: string;
  status: string;
  message?: string;
  description?: string;
  timestamp?: string;
}

export interface RegistrationRequest {
  email: string;
  password: string;
  name: string;
  phone_number: string;
  face_image: string;
  use_multi_sample?: boolean;
}

export interface RegistrationResponse {
  success: boolean;
  message: string;
  user_id?: string;
  email?: string;
  name?: string;
  samples_captured?: number;
  error?: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public data?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status}`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorData.message || errorMessage;
      throw new ApiError(errorMessage, response.status, errorData);
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw new ApiError(errorMessage, response.status);
    }
  }
  return response.json();
}

export const api = {
  faceRecognition: async (faceImageBase64: string): Promise<FaceRecognitionResponse> => {
    const response = await fetch('/api/face-recognition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ face_image: faceImageBase64 }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorDetail = errorData.error || errorData.detail || errorData.message || 'Face recognition failed';
      throw new ApiError(errorDetail, response.status, errorData);
    }
    return response.json();
  },

  processPayment: async (data: PaymentRequest): Promise<PaymentResponse> => {
    const response = await fetch('/api/mpesa/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorDetail = errorData.error || errorData.detail || errorData.message || 'Payment processing failed';
      throw new ApiError(errorDetail, response.status, errorData);
    }
    return response.json();
  },

  getTransactionStatus: async (transactionId: string): Promise<Transaction> => {
    const response = await fetch(`${BASE_URL}/mpesa/status/${transactionId}`);
    return handleResponse<Transaction>(response);
  },

  getUserTransactions: async (userId: string): Promise<{ transactions: Transaction[] }> => {
    const response = await fetch('/api/sales-transactions');
    return handleResponse<{ transactions: Transaction[] }>(response);
  },

  healthCheck: async (): Promise<{ status: string }> => {
    const response = await fetch(`${BASE_URL}/health`);
    return handleResponse<{ status: string }>(response);
  },

  registerUser: async (formData: FormData): Promise<RegistrationResponse> => {
    const response = await fetch('/api/registration', {
      method: 'POST',
      body: formData,
    });
    return handleResponse<RegistrationResponse>(response);
  },
};