import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { face_image } = body

    if (!face_image) {
      return NextResponse.json(
        { error: "face_image field is required" },
        { status: 400 }
      )
    }

    const response = await fetch("http://localhost:8000/face-recognition", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ face_image }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      const errorDetail = errorData.detail || errorData.message || "Face recognition failed"
      return NextResponse.json(
        { error: errorDetail },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Face recognition proxy error:", error)
    return NextResponse.json(
      { error: "Failed to connect to face recognition service" },
      { status: 500 }
    )
  }
}