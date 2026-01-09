"use client"

import { useState, useEffect, useCallback } from "react"
import { ScanFace, ArrowDownLeft, User, Phone, CheckCircle, Loader2, XCircle, Terminal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { CameraStream } from "@/components/camera-stream"
import { usePaymentStore } from "@/lib/store"
import { api } from "@/lib/api"

const quickAmounts = [100, 500, 1000]

type LogLevel = 'info' | 'success' | 'error' | 'warning'

interface LogEntry {
  id: string
  timestamp: string
  level: LogLevel
  message: string
}

export function HomePage() {
  const { amount, setAmount, status, setStatus, recognizedUser, recognitionScore, setRecognizedUser, setRecognitionScore, resetPayment, lastTransactionId } = usePaymentStore()
  const [error, setError] = useState<string | null>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [modelLoaded, setModelLoaded] = useState(false)
  const ipWebcamUrl = "http://192.168.1.117:8080/photo.jpg"

  const addLog = useCallback((level: LogLevel, message: string) => {
    const newLog: LogEntry = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
      level,
      message
    }
    setLogs(prev => [...prev.slice(-19), newLog])
  }, [])

  useEffect(() => {
    addLog('info', 'System initialized')
  }, [addLog])

  useEffect(() => {
    if (status === 'scanning') {
      addLog('info', 'Accessing camera...')
    } else if (status === 'processing') {
      addLog('warning', 'Detecting face...')
    } else if (status === 'recognized') {
      addLog('success', `Face detected: ${recognizedUser?.name || 'Unknown'}`)
    }
  }, [status, recognizedUser, addLog])

  useEffect(() => {
    const checkModel = async () => {
      try {
        await api.faceRecognition('')
        setModelLoaded(true)
        addLog('success', 'SFace model loaded')
      } catch (err) {
        setModelLoaded(false)
      }
    }
    checkModel()
  }, [addLog])

  const handleQuickAmount = (value: number) => {
    setAmount(value.toString())
  }

  const handleScan = async () => {
    if (!amount || parseFloat(amount) <= 0) {
      setError("Please enter a valid amount")
      return
    }
    setError(null)
    setStatus("scanning")
    setCapturedImage(null)
    addLog('info', `Camera connected - Scanning for KES ${parseFloat(amount).toFixed(2)}`)
  }

  const handleImageCapture = (imageData: string) => {
    if (status === "scanning") {
      setStatus("processing")
      setCapturedImage(imageData)
      recognizeFace(imageData)
    }
  }

  const recognizeFace = async (faceImage: string) => {
    try {
      const result = await api.faceRecognition(faceImage)
      if (result.success && result.user_id) {
        setRecognizedUser({
          user_id: result.user_id,
          name: result.name || "Unknown",
          phone_number: result.phone_number || "",
        })
        setRecognitionScore(result.confidence || 0)
        setStatus("recognized")
        const confidenceValue = result.confidence || 0
        const confidenceDisplay = confidenceValue > 1
          ? confidenceValue.toFixed(1) + '%'
          : (confidenceValue * 100).toFixed(1) + '%'
        addLog('success', `Face matched: ${result.name} (${confidenceDisplay})`)
      } else {
        addLog('warning', 'No face matched above 80% threshold')
        setStatus("idle")
        setError("No matching face found. Please try again.")
      }
    } catch (err) {
      addLog('error', 'Face recognition failed')
      setStatus("idle")
      setError("Failed to recognize face. Please try again.")
    }
  }

  const handlePayment = async () => {
    if (!recognizedUser) return

    setStatus("processing")
    setError(null)

    try {
      const result = await api.processPayment({
        amount: parseFloat(amount),
        phone_number: recognizedUser.phone_number,
        user_id: recognizedUser.user_id,
        description: "Face Recognition Payment",
      })
      
      setStatus("completed")
      alert(`Payment sent to ${recognizedUser.name}!\nCheck your phone for M-Pesa PIN.\nTransaction ID: ${result.transaction_id}`)
      resetPayment()
    } catch (err) {
      setError("Payment failed. Please try again.")
      setStatus("recognized")
    }
  }

  const handleCancel = () => {
    resetPayment()
    setCapturedImage(null)
    setError(null)
  }

  const formatCurrency = (value: number) => {
    return value.toLocaleString("en-KE", { minimumFractionDigits: 2 })
  }

  return (
    <div className="p-4 lg:p-8 max-w-4xl mx-auto space-y-6">
      <div className="bg-[#18181b] border border-[#27272a] rounded-2xl p-6 lg:p-8">
        <div className="text-center space-y-6">
          <div>
            <p className="text-sm font-medium text-[#a1a1aa] mb-2">Enter Amount</p>
            <div className="flex items-center justify-center gap-2">
              <span className="text-3xl font-mono font-bold text-[#a1a1aa]">KES</span>
              <Input
                type="number"
                placeholder="0.00"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                disabled={status !== "idle"}
                className="text-5xl lg:text-6xl font-mono font-bold text-center border-0 bg-transparent focus-visible:ring-0 w-48 lg:w-64 p-0 placeholder:text-[#3f3f46] text-white disabled:opacity-50"
              />
            </div>
          </div>

          <div className="flex items-center justify-center gap-3">
            {quickAmounts.map((value) => (
              <Button
                key={value}
                variant="outline"
                size="sm"
                disabled={status !== "idle"}
                onClick={() => handleQuickAmount(value)}
                className={`px-6 py-2 rounded-full font-mono font-medium transition-all border disabled:opacity-50 ${
                  amount === value.toString()
                    ? "bg-[#ccff00] text-black border-[#ccff00] hover:bg-[#b8e600] hover:text-black"
                    : "border-[#27272a] text-[#a1a1aa] bg-transparent hover:border-[#ccff00] hover:text-[#ccff00]"
                }`}
              >
                {value.toLocaleString()}
              </Button>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-[#18181b] border border-[#27272a] rounded-2xl overflow-hidden">
        <div className={`relative ${status === "idle" ? "aspect-[4/3] lg:aspect-video" : "aspect-[4/3] lg:aspect-video"}`}>
          <CameraStream
            isActive={status !== "processing" && status !== "completed"}
            onCapture={handleImageCapture}
            className="w-full h-full"
            ipWebcamUrl={ipWebcamUrl}
          rotation={180}
            />

          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div
              className={`relative w-48 h-56 lg:w-56 lg:h-64 border-2 border-dashed rounded-3xl transition-all duration-300 ${
                status === "scanning" ? "border-[#ccff00]" : status === "recognized" ? "border-[#ccff00]" : "border-[#3f3f46]"
              }`}
            >
              <div className="absolute -top-1 -left-1 w-6 h-6 border-t-2 border-l-2 border-[#ccff00] rounded-tl-xl" />
              <div className="absolute -top-1 -right-1 w-6 h-6 border-t-2 border-r-2 border-[#ccff00] rounded-tr-xl" />
              <div className="absolute -bottom-1 -left-1 w-6 h-6 border-b-2 border-l-2 border-[#ccff00] rounded-bl-xl" />
              <div className="absolute -bottom-1 -right-1 w-6 h-6 border-b-2 border-r-2 border-[#ccff00] rounded-br-xl" />
            </div>
          </div>

          <div className="absolute top-4 left-4 right-4 flex items-center justify-between">
            <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-full border border-[#27272a]">
              <div className={`w-2 h-2 rounded-full ${
                status === "scanning" ? "bg-[#ccff00] animate-pulse" : status === "processing" ? "bg-yellow-500 animate-pulse" : "bg-[#ccff00]"
              }`} />
              <span className="text-white/90 text-xs font-medium">
                {status === "idle" ? "Ready" : status === "scanning" ? "Scanning..." : status === "processing" ? "Processing..." : status === "recognized" ? "User Found" : "Completed"}
              </span>
            </div>
            {amount && (
              <div className="bg-[#ccff00] backdrop-blur-sm px-3 py-1.5 rounded-full">
                <span className="text-black text-xs font-mono font-semibold">
                  KES {Number.parseFloat(amount || "0").toLocaleString("en-KE", { minimumFractionDigits: 2 })}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {status === "idle" && (
        <Button
          onClick={handleScan}
          disabled={!amount}
          className="w-full h-14 text-lg font-semibold bg-[#ccff00] hover:bg-[#b8e600] text-black rounded-2xl transition-all disabled:opacity-50 disabled:cursor-not-allowed border-0"
        >
          <ScanFace className="h-6 w-6 mr-2" />
          Scan Face to Pay
        </Button>
      )}

      {status === "recognized" && recognizedUser && (
        <div className="bg-[#18181b] border-2 border-[#ccff00] rounded-2xl p-6 space-y-4">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-full bg-[#ccff00]/10 flex items-center justify-center border border-[#ccff00]">
              <User className="h-7 w-7 text-[#ccff00]" />
            </div>
            <div className="flex-1">
              <p className="text-[#ccff00] text-xs font-mono">USER IDENTIFIED</p>
              <p className="text-white text-lg font-semibold">{recognizedUser.name}</p>
              <p className="text-[#a1a1aa] text-sm font-mono">{recognizedUser.phone_number}</p>
              <p className="text-[#ccff00] text-sm font-mono mt-1">
                Confidence: {recognitionScore.toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="border-t border-[#27272a] pt-4">
            <p className="text-[#a1a1aa] text-sm mb-1">Payment Amount</p>
            <p className="text-[#ccff00] text-3xl font-semibold font-mono">
              KES {formatCurrency(parseFloat(amount || "0"))}
            </p>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <Button
              onClick={() => {
                setRecognizedUser(null)
                setRecognitionScore(0)
                setStatus("idle")
                setCapturedImage(null)
                addLog('info', 'User mismatch - ready to rescan')
              }}
              variant="ghost"
              className="h-12 text-[#a1a1aa] hover:!bg-red-500 hover:!border-red-500 hover:!text-white border border-transparent"
            >
              <ArrowDownLeft className="h-4 w-4 mr-2" />
              Not You?
            </Button>
            <Button
              onClick={handlePayment}
              disabled={status === "processing"}
              className="h-12 bg-[#ccff00] hover:bg-[#b8e600] text-black font-semibold"
            >
              {status === "processing" ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Processing
                </>
              ) : (
                <>
                  <CheckCircle className="h-5 w-5 mr-2" />
                  Confirm Payment
                </>
              )}
            </Button>
            <Button
              onClick={handleCancel}
              variant="outline"
              className="h-12 border-[#27272a] text-[#a1a1aa] hover:border-red-500 hover:text-red-500"
            >
              <XCircle className="h-5 w-5 mr-2" />
              Cancel Payment
            </Button>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
          <XCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
          <p className="text-red-400 text-sm">{error}</p>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCancel}
            className="ml-auto text-red-400 hover:text-red-300"
          >
            Try Again
          </Button>
        </div>
      )}

      <div className="bg-[#18181b] border border-[#27272a] rounded-2xl">
        <div className="p-4 border-b border-[#27272a] flex items-center gap-2">
          <Terminal className="h-5 w-5 text-[#ccff00]" />
          <h2 className="text-lg font-semibold text-white">System Logs</h2>
        </div>
        <div className="p-4 space-y-2 max-h-64 overflow-y-auto">
          {logs.length === 0 ? (
            <p className="text-[#52525b] text-sm text-center py-4">No logs yet</p>
          ) : (
            logs.map((log) => (
              <div key={log.id} className="flex items-start gap-2 text-xs font-mono">
                <span className="text-[#52525b] flex-shrink-0 mt-0.5">[{log.timestamp}]</span>
                <span className={`flex-shrink-0 mt-0.5 ${
                  log.level === 'success' ? 'text-green-500' :
                  log.level === 'error' ? 'text-red-500' :
                  log.level === 'warning' ? 'text-yellow-500' :
                  'text-blue-400'
                }`}>
                  {log.level === 'success' ? '✅' :
                   log.level === 'error' ? '❌' :
                   log.level === 'warning' ? '⚠️' :
                   'ℹ️'}
                </span>
                <span className={`${
                  log.level === 'success' ? 'text-green-400' :
                  log.level === 'error' ? 'text-red-400' :
                  log.level === 'warning' ? 'text-yellow-400' :
                  'text-blue-300'
                }`}>
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="bg-[#18181b] border border-[#27272a] rounded-2xl">
        <div className="p-4 border-b border-[#27272a]">
          <h2 className="text-lg font-semibold text-white">Recent Sales</h2>
        </div>
        <div className="p-2 text-center py-8">
          <p className="text-[#52525b] text-sm">View Sales History page for transactions</p>
        </div>
      </div>
    </div>
  )
}

export default HomePage