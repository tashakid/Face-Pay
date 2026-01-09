import { NextResponse } from 'next/server'

export async function GET(
  request: Request,
  { params }: { params: { userId: string } }
) {
  try {
    const { userId } = params

    const response = await fetch(`http://localhost:8000/mpesa/transactions/${userId}`, {
      cache: 'no-store',
    })

    if (!response.ok) {
      console.error('Backend returned status:', response.status)
      return NextResponse.json(
        { transactions: [] },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Transactions API proxy error:', error)
    return NextResponse.json(
      { transactions: [] },
      { status: 503 }
    )
  }
}