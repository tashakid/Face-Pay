import { NextResponse } from 'next/server'

export async function GET(
  request: Request,
  { params }: { params: { slug: string[] } }
) {
  try {
    const url = new URL(request.url)
    const pathSegments = url.pathname.split('/')
    const lastSegment = pathSegments[pathSegments.length - 1]

    if (lastSegment === 'all') {
      const response = await fetch('http://localhost:8000/mpesa/transactions/all', {
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
    }

    return NextResponse.json(
      { error: 'Invalid endpoint' },
      { status: 400 }
    )
  } catch (error) {
    console.error('Transactions API proxy error:', error)
    return NextResponse.json(
      { transactions: [] },
      { status: 503 }
    )
  }
}