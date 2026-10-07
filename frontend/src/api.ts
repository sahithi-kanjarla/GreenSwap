import type { AgentEvent, AgentResult, DemoSummary, Offer } from './types'

export interface SearchOptions {
  preferences: string[]
  maxPrice: number | null
  sortBy: string
  replay: boolean
}

/** Streams agent progress over Server-Sent Events. Returns a function that cancels the stream. */
export function streamSearch(query: string, opts: SearchOptions, onEvent: (e: AgentEvent) => void): () => void {
  const params = new URLSearchParams({ q: query, sort_by: opts.sortBy })
  if (opts.preferences.length) params.set('preferences', opts.preferences.join(','))
  if (opts.maxPrice) params.set('max_price', String(opts.maxPrice))
  if (opts.replay) params.set('replay', '1')

  const source = new EventSource(`/api/search?${params}`)
  let finished = false
  source.onmessage = (msg) => {
    const event = JSON.parse(msg.data) as AgentEvent
    if (event.type === 'result' || event.type === 'error') {
      finished = true
      source.close()
    }
    onEvent(event)
  }
  source.onerror = () => {
    if (!finished) onEvent({ type: 'error', message: 'Lost connection to the GreenSwap server.' })
    source.close()
  }
  return () => source.close()
}

export async function rerank(body: { run_id: string; preferences: string[]; max_price: number | null; sort_by: string }): Promise<AgentResult> {
  const res = await fetch('/api/rank', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!res.ok) throw new Error((await res.json()).detail ?? 'Re-ranking failed')
  return res.json()
}

export async function fetchOffers(token: string): Promise<Offer[]> {
  const res = await fetch(`/api/offers?token=${encodeURIComponent(token)}`)
  if (!res.ok) throw new Error((await res.json()).detail ?? 'Could not load store offers')
  return (await res.json()).offers
}

export async function fetchDemos(): Promise<DemoSummary[]> {
  const res = await fetch('/api/demos')
  return res.ok ? res.json() : []
}
