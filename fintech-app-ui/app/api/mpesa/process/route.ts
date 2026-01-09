import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await fetch("http://localhost:8000/mpesa/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      const errorDetail = errorData.detail || errorData.message || "Payment processing failed"
      return NextResponse.json(
        { error: errorDetail },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Payment proxy error:", error)
    return NextResponse.json(
      { error: "Failed to connect to payment service" },
      { status: 500 }
    )
  }
}