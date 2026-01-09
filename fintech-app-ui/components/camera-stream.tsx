"use client"

import { useEffect, useRef, useState } from "react"
import { Camera, CameraOff, RefreshCw } from "lucide-react"

interface CameraStreamProps {
  isActive?: boolean
  onCapture?: (imageData: string) => void
  className?: string
  ipWebcamUrl?: string
  rotation?: number
}

export function CameraStream({ isActive = false, onCapture, className = "", ipWebcamUrl, rotation = 0 }: CameraStreamProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const onCaptureRef = useRef(onCapture)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [cameraError, setCameraError] = useState<Error | null>(null)
  const [usingIpCamera, setUsingIpCamera] = useState(!!ipWebcamUrl)
  const [retryCount, setRetryCount] = useState(0)
  const [lastErrorDetails, setLastErrorDetails] = useState<string | null>(null)
  const [displayFrame, setDisplayFrame] = useState<string | null>(null)

  useEffect(() => {
    onCaptureRef.current = onCapture
  }, [onCapture])

  const getErrorMessage = (err: Error): string => {
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      return "Camera access denied. Please allow camera permissions in your browser."
    }
    if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      return "No camera found. Please connect a camera or check IP Webcam URL."
    }
    if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      return "Camera is already in use by another application."
    }
    if (err.name === "OverconstrainedError" || err.name === "ConstraintNotSatisfiedError") {
      return "Camera doesn't support the requested settings."
    }
    if (err.name === "TypeError" && err.message.includes("getUserMedia")) {
      return "Camera not available. Your browser may not support camera access, or you may need to use HTTPS."
    }
    return `Camera error: ${err.message}`
  }

  useEffect(() => {
    if (!isActive) {
      setIsReady(false)
      if (videoRef.current) {
        videoRef.current.src = ""
      }
      return
    }

    let mounted = true

    if (ipWebcamUrl) {
      setUsingIpCamera(true)
      setError(null)

      const proxyUrl = `/api/video-proxy?url=${encodeURIComponent(ipWebcamUrl)}`
      console.log("📡 Using IP Webcam in image mode:", proxyUrl)

      let isLoading = false

      const loadFrame = async () => {
        if (!mounted || isLoading) return
        
        isLoading = true

        try {
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout

          const response = await fetch(proxyUrl + "&t=" + Date.now(), {
            signal: controller.signal
          })
          
          clearTimeout(timeoutId)

          if (!response.ok) {
            if (response.status === 429) {
              console.warn("⚠️ Rate limited - skipping frame")
              return
            }
            console.warn("⚠️ Frame fetch failed:", response.status)
            return
          }

          const blob = await response.blob()
          
          if (blob.size === 0 || blob.size < 1000) {
            console.warn("⚠️ Invalid frame size:", blob.size)
            return
          }

          const bitmap = await createImageBitmap(blob).catch(() => null)

          if (!bitmap) {
            console.warn("⚠️ Could not decode image")
            return
          }

          if (!isReady) {
            console.log("✅ IP Webcam ready, resolution:", bitmap.width, "x", bitmap.height)
            setIsReady(true)
            setCameraError(null)
          }

          if (mounted) {
            const canvas = canvasRef.current || document.createElement("canvas")

            if (bitmap.width > bitmap.height || rotation !== 0) {
              if (rotation === 0) {
                canvas.width = bitmap.height
                canvas.height = bitmap.width
              } else {
                canvas.width = bitmap.width
                canvas.height = bitmap.height
              }

              const ctx = canvas.getContext("2d")

              if (ctx) {
                ctx.save()

                if (rotation === 0 && bitmap.width > bitmap.height) {
                  ctx.translate(canvas.width / 2, canvas.height / 2)
                  ctx.rotate(-90 * Math.PI / 180)
                  ctx.drawImage(bitmap, -bitmap.width / 2, -bitmap.height / 2)
                } else if (rotation !== 0) {
                  ctx.translate(canvas.width / 2, canvas.height / 2)
                  ctx.rotate(rotation * Math.PI / 180)
                  ctx.drawImage(bitmap, -bitmap.width / 2, -bitmap.height / 2)
                } else {
                  ctx.drawImage(bitmap, 0, 0)
                }

                ctx.restore()
              }
            } else {
              canvas.width = bitmap.width
              canvas.height = bitmap.height
              const ctx = canvas.getContext("2d")

              if (ctx) {
                ctx.drawImage(bitmap, 0, 0)
              }
            }

            const dataUrl = canvas.toDataURL("image/jpeg", 0.85)
            canvasRef.current = canvas

            if (mounted) {
              setDisplayFrame(dataUrl)
            }

            bitmap.close()
          }
        } catch (err) {
          const error = err as Error
          if (error.name === 'AbortError') {
            console.warn("⚠️ Frame timeout")
          } else {
            console.warn("⚠️ Frame load error:", err)
            if (mounted && !isReady) {
              setError(`Failed to load camera image: ${error.message}`)
            }
          }
        } finally {
          isLoading = false
        }
      }

      loadFrame()
      const interval = setInterval(() => {
        if (mounted && !isLoading) {
          loadFrame()
        }
      }, 300) // Reduced from 30ms to 300ms (3.3 FPS instead of 33 FPS)

      if (videoRef.current) {
        (videoRef.current as any).intervalId = interval
      }
    } else {
      const startLocalCamera = async () => {
        try {
          const constraints = {
            video: {
              facingMode: "user",
              width: { ideal: 1280 },
              height: { ideal: 720 }
            },
            audio: false
          }

          console.log("📷 Starting local camera")
          const mediaStream = await navigator.mediaDevices.getUserMedia(constraints)

          if (!mounted) return

          setStream(mediaStream)
          setError(null)
          setCameraError(null)

          if (videoRef.current) {
            videoRef.current.srcObject = mediaStream
            videoRef.current.onloadedmetadata = () => {
              if (!videoRef.current) return

              videoRef.current.play().then(() => {
                console.log("✅ Local camera playing")
                setIsReady(true)
              }).catch(playErr => {
                console.error("❌ Local camera play error:", playErr)
                if (mounted) {
                  setError("Failed to play camera stream: " + playErr.message)
                  setCameraError(playErr as Error)
                }
              })
            }

            videoRef.current.onerror = (e) => {
              console.error("❌ Local camera error:", e)
              if (mounted) {
                setError("Camera stream error")
                setCameraError(new Error("Video element error occurred"))
              }
            }
          }
        } catch (err) {
          if (!mounted) return

          const errorObj = err instanceof Error ? err : new Error(String(err))
          console.error("❌ Camera error:", errorObj)

          setError(getErrorMessage(errorObj))
          setCameraError(errorObj)
        }
      }

      startLocalCamera()
    }

    return () => {
      mounted = false
      if (videoRef.current) {
        const intervalId = (videoRef.current as any).intervalId
        if (intervalId) {
          clearInterval(intervalId)
        }
        if (usingIpCamera) {
          videoRef.current.src = ""
        }
      }
      if (stream) {
        stream.getTracks().forEach(track => {
          track.stop()
          console.log("🛑 Stopped camera track:", track.kind)
        })
      }
    }
  }, [isActive, ipWebcamUrl])

  const captureFrame = (): string | null => {
    if (usingIpCamera) {
      const canvas = canvasRef.current
      if (!canvas || !isReady) return null

      try {
        return canvas.toDataURL("image/jpeg", 0.85)
      } catch (err) {
        console.error("❌ Capture frame error:", err)
        return null
      }
    }

    if (!videoRef.current || !isReady || !videoRef.current.videoWidth) {
      return null
    }

    try {
      const canvas = canvasRef.current || document.createElement("canvas")
      if (!canvasRef.current) {
        canvasRef.current = canvas
      }

      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext("2d")

      if (!ctx) return null

      ctx.drawImage(videoRef.current, 0, 0)

      return canvas.toDataURL("image/jpeg", 0.85)
    } catch (err) {
      console.error("❌ Capture frame error:", err)
      return null
    }
  }

  useEffect(() => {
    if (isActive && isReady && onCaptureRef.current) {
      const interval = setInterval(() => {
        const imageData = captureFrame()
        if (imageData && onCaptureRef.current) {
          onCaptureRef.current(imageData)
        }
      }, 500) // Capture interval for sending to recognition endpoint

      return () => clearInterval(interval)
    }
  }, [isActive, isReady])

  const tryDirectVideo = () => {
    setError(null)
    setRetryCount(0)
    setLastErrorDetails(null)
    console.log("🔄 Retry camera connection")
    if (ipWebcamUrl && videoRef.current) {
      videoRef.current.src = `/api/video-proxy?url=${encodeURIComponent(ipWebcamUrl)}`
    }
  }

  if (error) {
    return (
      <div className={`flex flex-col items-center justify-center bg-[#09090b] ${className}`}>
        <CameraOff className="h-12 w-12 text-red-500 mb-2" />
        <p className="text-[#a1a1aa] text-sm text-center px-4">{error}</p>
        {ipWebcamUrl && (
          <button
            onClick={tryDirectVideo}
            className="mt-4 flex items-center gap-2 text-[#ccff00] text-sm hover:underline"
          >
            <RefreshCw className="h-4 w-4" />
            Retry
          </button>
        )}
        <div className="mt-4 text-xs text-left px-4 text-[#52525b]">
          <p className="mb-1">
            {cameraError?.name === "NotAllowedError" && "Click the camera icon in your browser address bar to allow access"}
            {cameraError?.name === "NotFoundError" && usingIpCamera && "Ensure IP Webcam app is running on your phone"}
          </p>
          {lastErrorDetails && (
            <details className="mt-2">
              <summary className="cursor-pointer text-[#a1a1aa] hover:text-[#ccff00]">
                Error Details (click to expand)
              </summary>
              <pre className="mt-2 p-2 bg-[#18181b] rounded text-[10px] overflow-auto max-h-32 text-red-400">
                {lastErrorDetails}
              </pre>
            </details>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className={`relative bg-[#09090b] overflow-hidden ${className}`}>
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#09090b]">
          <Camera className="h-12 w-12 text-[#ccff00] animate-pulse" />
        </div>
      )}

      {usingIpCamera && isReady && displayFrame && (
        <img
          src={displayFrame}
          alt="IP Webcam"
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
        />
      )}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        crossOrigin="anonymous"
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${isReady && !usingIpCamera ? "opacity-100" : "opacity-0"}`}
        style={{ transform: "scaleX(-1)" }}
      />

      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

export default CameraStream
