import { create } from 'zustand';
import type { Transaction } from './api';

interface RecognizedUser {
  user_id: string;
  name: string;
  phone_number: string;
  email?: string;
}

interface PaymentState {
  amount: string;
  setAmount: (amount: string) => void;

  status: 'idle' | 'scanning' | 'recognized' | 'processing' | 'completed' | 'failed';
  setStatus: (status: PaymentState['status']) => void;

  recognizedUser: RecognizedUser | null;
  setRecognizedUser: (user: RecognizedUser | null) => void;

  recognitionScore: number;
  setRecognitionScore: (score: number) => void;

  lastTransactionId: string | null;
  setLastTransactionId: (id: string | null) => void;

  resetPayment: () => void;
}

interface CameraState {
  stream: MediaStream | null;
  setStream: (stream: MediaStream | null) => void;
  error: string | null;
  setError: (error: string | null) => void;
  isReady: boolean;
  setIsReady: (ready: boolean) => void;
  clearCamera: () => void;
}

interface TransactionState {
  transactions: Transaction[];
  setTransactions: (transactions: Transaction[]) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  fetchTransactions: (userId: string) => Promise<void>;
}

interface AppState {
  payment: PaymentState;
  camera: CameraState;
  transactions: TransactionState;
}

export const useStore = create<AppState>((set, get) => ({
  payment: {
    amount: '',
    setAmount: (amount) => set({ payment: { ...get().payment, amount } }),

    status: 'idle',
    setStatus: (status) => set({ payment: { ...get().payment, status } }),

    recognizedUser: null,
    setRecognizedUser: (user) => set({ payment: { ...get().payment, recognizedUser: user } }),

    recognitionScore: 0,
    setRecognitionScore: (score) => set({ payment: { ...get().payment, recognitionScore: score } }),

    lastTransactionId: null,
    setLastTransactionId: (id) => set({ payment: { ...get().payment, lastTransactionId: id } }),

    resetPayment: () =>
      set({
        payment: {
          amount: '',
          status: 'idle',
          recognizedUser: null,
          recognitionScore: 0,
          lastTransactionId: null,
          setAmount: get().payment.setAmount,
          setStatus: get().payment.setStatus,
          setRecognizedUser: get().payment.setRecognizedUser,
          setRecognitionScore: get().payment.setRecognitionScore,
          setLastTransactionId: get().payment.setLastTransactionId,
          resetPayment: get().payment.resetPayment,
        },
      }),
  },

  camera: {
    stream: null,
    setStream: (stream) => set({ camera: { ...get().camera, stream } }),
    error: null,
    setError: (error) => set({ camera: { ...get().camera, error } }),
    isReady: false,
    setIsReady: (ready) => set({ camera: { ...get().camera, isReady: ready } }),
    clearCamera: () =>
      set({
        camera: {
          stream: null,
          error: null,
          isReady: false,
          setStream: get().camera.setStream,
          setError: get().camera.setError,
          setIsReady: get().camera.setIsReady,
          clearCamera: get().camera.clearCamera,
        },
      }),
  },

  transactions: {
    transactions: [],
    setTransactions: (transactions) => set({ transactions: { ...get().transactions, transactions } }),
    isLoading: false,
    setIsLoading: (loading) => set({ transactions: { ...get().transactions, isLoading: loading } }),
    fetchTransactions: async (userId: string) => {
      const { setIsLoading, setTransactions } = get().transactions;
      const { api } = await import('./api');

      setIsLoading(true);
      try {
        const response = await api.getUserTransactions(userId);
        setTransactions(response.transactions);
      } catch (error) {
        console.error('Failed to fetch transactions:', error);
      } finally {
        setIsLoading(false);
      }
    },
  },
}));

export const usePaymentStore = () => useStore((state) => state.payment);
export const useCameraStore = () => useStore((state) => state.camera);
export const useTransactionStore = () => useStore((state) => state.transactions);