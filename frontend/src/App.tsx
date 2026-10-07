import { useEffect, useRef, useState } from 'react'
import type { AgentEvent, AgentResult, DemoSummary } from './types'
import { fetchDemos, rerank, streamSearch } from './api'
import { AgentTimeline } from './components/AgentTimeline'
import { Results } from './components/Results'

const EXAMPLES = [
  'virgin plastic chair',
  'water bottle for office',
  'floor cleaner for a home with a baby',
  'lunch box for carrying food to college',
]

const SORTS: [string, string][] = [
  ['recommended', 'Recommended'],
  ['price_low_high', 'Price: low to high'],
  ['price_high_low', 'Price: high to low'],
  ['rating', 'Rating'],
  ['reviews', 'Most reviewed'],
]

export default function App() {
  const [query, setQuery] = useState('')
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<AgentResult | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [demos, setDemos] = useState<DemoSummary[]>([])
  const [lowCost, setLowCost] = useState(false)
  const [maxPrice, setMaxPrice] = useState('')
  const [sortBy, setSortBy] = useState('recommended')
  const cancel = useRef<() => void>(() => {})

  useEffect(() => {
    fetchDemos().then(setDemos).catch(() => setDemos([]))
    return () => cancel.current()
  }, [])

  const prefs = lowCost ? ['low_cost'] : []
  const budget = maxPrice ? Number(maxPrice) : null

  function start(q: string, replay = false) {
    const text = q.trim()
    if (!text || running) return
    cancel.current()
    setQuery(replay ? (demos.find((d) => d.slug === text)?.query ?? text) : text)
    setEvents([])
    setResult(null)
    setRunId(null)
    setError('')
    setRunning(true)
    cancel.current = streamSearch(text, { preferences: prefs, maxPrice: budget, sortBy, replay }, (e) => {
      if (e.type === 'result') {
        setResult(e.view)
        setRunId(e.run_id)
        setRunning(false)
      } else if (e.type === 'error') {
        setError(e.message)
        setRunning(false)
      } else {
        setEvents((prev) => [...prev, e])
      }
    })
  }

  // Sorting, budget and preferences re-rank the finished run in pure code:
  // no new searches, no LLM calls.
  useEffect(() => {
    if (!runId) return
    rerank({ run_id: runId, preferences: prefs, max_price: budget, sort_by: sortBy })
      .then(setResult)
      .catch((e: Error) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortBy, lowCost, maxPrice])

  return (
    <div className="mx-auto max-w-6xl px-4 py-6 sm:py-10">
      <header className="mb-6 flex items-center gap-3">
        <img src="/favicon.svg" alt="" className="h-9 w-9" />
        <div>
          <h1 className="text-2xl font-bold tracking-tight">GreenSwap</h1>
          <p className="text-sm text-stone-500">Before you buy: better alternatives as real products, with honest evidence.</p>
        </div>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          start(query)
        }}
        className="space-y-3"
      >
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What are you about to buy? A product, a need, or a product URL"
            aria-label="Product, need or product URL"
            className="min-w-0 flex-1 rounded-xl border border-stone-300 bg-white px-4 py-3 text-base outline-none focus:border-leaf-500 focus:ring-2 focus:ring-leaf-500/30 dark:border-stone-700 dark:bg-stone-900"
          />
          <button
            type="submit"
            disabled={running || !query.trim()}
            className="rounded-xl bg-leaf-600 px-5 py-3 font-semibold text-white hover:bg-leaf-700 disabled:opacity-50"
          >
            {running ? 'Searching…' : 'Find alternatives'}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={lowCost} onChange={(e) => setLowCost(e.target.checked)} className="accent-leaf-600" />
            Prefer lower cost
          </label>
          <label className="flex items-center gap-1.5">
            Budget ₹
            <input
              type="number"
              min={0}
              inputMode="numeric"
              value={maxPrice}
              onChange={(e) => setMaxPrice(e.target.value)}
              placeholder="any"
              className="w-24 rounded-lg border border-stone-300 bg-white px-2 py-1 dark:border-stone-700 dark:bg-stone-900"
            />
          </label>
          <label className="flex items-center gap-1.5">
            Sort
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="rounded-lg border border-stone-300 bg-white px-2 py-1 dark:border-stone-700 dark:bg-stone-900"
            >
              {SORTS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
            </select>
          </label>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="py-1 text-stone-500">Try:</span>
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" onClick={() => start(ex)} disabled={running}
                    className="rounded-full border border-stone-300 px-3 py-1 hover:border-leaf-500 disabled:opacity-50 dark:border-stone-700">
              {ex}
            </button>
          ))}
        </div>
        {demos.length > 0 && (
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="py-1 text-stone-500">Replay a saved run (no credits):</span>
            {demos.map((d) => (
              <button key={d.slug} type="button" onClick={() => start(d.slug, true)} disabled={running}
                      className="rounded-full bg-stone-200 px-3 py-1 hover:bg-stone-300 disabled:opacity-50 dark:bg-stone-800 dark:hover:bg-stone-700">
                {d.query}
              </button>
            ))}
          </div>
        )}
      </form>

      <main className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="order-2 min-w-0 lg:order-1">
          {error && <p role="alert" className="mb-4 rounded-xl border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">{error}</p>}
          {result ? (
            <Results result={result} />
          ) : running ? (
            <p className="text-sm text-stone-500">The agent is planning searches, reading real listings and checking its own claims. This usually takes 20–60 seconds.</p>
          ) : (
            !error && (
              <div className="rounded-2xl border border-dashed border-stone-300 p-6 text-sm text-stone-600 dark:border-stone-700 dark:text-stone-400">
                <p className="mb-2 font-medium text-stone-800 dark:text-stone-200">How GreenSwap stays honest</p>
                <ul className="list-disc space-y-1 pl-5">
                  <li>The AI plans the searches; code caps them and only shows products that really came back from Google Shopping.</li>
                  <li>Every claim is checked against text we actually retrieved and labelled <i>Evidence found</i>, <i>Stated by seller</i> or <i>Not verified</i>.</li>
                  <li>No made-up "eco score". Popularity and offers are buying signals, never environmental proof.</li>
                </ul>
              </div>
            )
          )}
        </div>
        <aside className="order-1 lg:order-2 lg:sticky lg:top-6 lg:self-start">
          <AgentTimeline events={events} running={running} />
        </aside>
      </main>
    </div>
  )
}
