export const dynamic = 'force-dynamic'

// Simple in-memory cache for last frame
const cache = new Map<string, { data: Buffer, timestamp: number }>()
const CACHE_TTL = 150 // ms - max age of cached frame

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const url = searchParams.get('url')
  const timestamp = searchParams.get('t')

  if (!url) {
    return new Response('Missing URL parameter', { status: 400 })
  }

  try {
    // Check cache first
    const cacheKey = url
    const cached = cache.get(cacheKey)
    const now = Date.now()
    
    if (cached && (now - cached.timestamp) < CACHE_TTL) {
      console.log('✅ Serving cached frame')
      return new Response(cached.data, {
        headers: {
          'Content-Type': 'image/jpeg',
          'Cache-Control': 'private, max-age=0.1',
          'X-Cache': 'HIT',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      })
    }

    // Fetch from camera
    const response = await fetch(url, {
      signal: AbortSignal.timeout(5000) // 5 seconds timeout
    })

    if (!response.ok) {
      return new Response('Failed to fetch video stream', {
        status: response.status,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
          'X-Cache': 'MISS',
        },
      })
    }

    const buffer = Buffer.from(await response.arrayBuffer())
    
    // Validate frame size
    if (buffer.length < 1000) {
      console.warn('⚠️ Invalid frame size:', buffer.length)
      return new Response('Invalid frame', {
        status: 422,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      })
    }

    // Update cache
    cache.set(cacheKey, { data: buffer, timestamp: now })
    
    // Clean old cache entries
    if (cache.size > 10) {
      const oldestKey = Array.from(cache.entries())[0][0]
      cache.delete(oldestKey)
    }

    return new Response(buffer, {
      headers: {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'private, max-age=0.1',
        'X-Cache': 'MISS',
        'X-Frame-Size': buffer.length.toString(),
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Expose-Headers': 'X-Cache, X-Frame-Size',
      },
    })
  } catch (error) {
    console.error('Proxy error:', error)
    
    // Return cached frame if available on error
    const cacheKey = url
    const cached = cache.get(cacheKey)
    if (cached && (Date.now() - cached.timestamp) < 3000) { // 3 second max age on error
      console.log('✅ Serving stale cached frame due to error')
      return new Response(cached.data, {
        headers: {
          'Content-Type': 'image/jpeg',
          'Cache-Control': 'private, max-age=0.1',
          'X-Cache': 'STALE',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      })
    }
    
    return new Response('Proxy error: ' + (error as Error).message, {
      status: 500,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    })
  }
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
