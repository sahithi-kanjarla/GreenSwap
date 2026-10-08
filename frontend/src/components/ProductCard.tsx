import { useState } from 'react'
import type { Claim, Offer, Product } from '../types'
import { fetchOffers } from '../api'

const CLAIM_STYLE: Record<Claim['status'], string> = {
  supported_by_search: 'bg-leaf-100 text-leaf-700 dark:bg-leaf-900 dark:text-leaf-100',
  stated_in_listing: 'bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-200',
  general_evidence: 'bg-violet-100 text-violet-800 dark:bg-violet-950 dark:text-violet-200',
  conflicting_evidence: 'bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200',
  unverified: 'bg-stone-200 text-stone-700 dark:bg-stone-800 dark:text-stone-300',
}

function ClaimChip({ claim }: { claim: Claim }) {
  const tip = claim.counter_snippet
    ? `Claimed: “${claim.evidence_snippet}” but also found: “${claim.counter_snippet}”`
    : claim.reason
      ? `${claim.evidence_snippet} (${claim.reason})`
      : `“${claim.evidence_snippet}”${claim.source_title ? ` from ${claim.source_title}` : ''}`
  return (
    <li className="text-xs">
      <span className={`mr-1.5 inline-block rounded px-1.5 py-0.5 font-medium ${CLAIM_STYLE[claim.status]}`} title={tip}>
        {claim.label}
      </span>
      <span className="text-stone-700 dark:text-stone-300">{claim.claim}</span>
      {claim.source_url && claim.status !== 'stated_in_listing' && (
        <a href={claim.source_url} target="_blank" rel="noreferrer" className="ml-1 text-leaf-600 underline dark:text-leaf-500">
          source
        </a>
      )}
    </li>
  )
}

function Offers({ token }: { token: string }) {
  const [state, setState] = useState<'idle' | 'loading' | 'error' | 'done'>('idle')
  const [offers, setOffers] = useState<Offer[]>([])
  const [error, setError] = useState('')

  async function load() {
    setState('loading')
    try {
      setOffers(await fetchOffers(token))
      setState('done')
    } catch (e) {
      setError((e as Error).message)
      setState('error')
    }
  }

  if (state === 'idle')
    return (
      <button onClick={load} className="text-xs font-medium text-leaf-600 hover:underline dark:text-leaf-500">
        Compare stores →
      </button>
    )
  if (state === 'loading') return <p className="text-xs text-stone-500">Checking other sellers…</p>
  if (state === 'error') return <p className="text-xs text-red-600">{error}</p>
  if (!offers.length) return <p className="text-xs text-stone-500">No other sellers found for this item.</p>
  return (
    <ul className="mt-1 w-full divide-y divide-stone-200 rounded-lg border border-stone-200 text-xs dark:divide-stone-800 dark:border-stone-800">
      {offers.map((o, i) => (
        <li key={i} className="flex items-center justify-between gap-2 px-2 py-1.5">
          <span className="truncate">
            {o.merchant}
            {o.tag ? <span className="ml-1 text-stone-500">· {o.tag}</span> : null}
          </span>
          <span className="flex shrink-0 items-center gap-2">
            <span className="font-semibold">{o.total ?? o.price ?? '—'}</span>
            {o.link && (
              <a href={o.link} target="_blank" rel="noreferrer" className="text-leaf-600 underline dark:text-leaf-500">
                visit
              </a>
            )}
          </span>
        </li>
      ))}
    </ul>
  )
}

export function ProductCard({ product }: { product: Product }) {
  const badges = product.shopping_signals?.badges ?? []
  return (
    <article className="flex flex-col overflow-hidden rounded-2xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
      <div className="flex gap-3 p-3">
        {product.image ? (
          <img src={product.image} alt="" loading="lazy" className="h-24 w-24 shrink-0 rounded-lg bg-white object-contain" />
        ) : (
          <div className="h-24 w-24 shrink-0 rounded-lg bg-stone-100 dark:bg-stone-800" />
        )}
        <div className="min-w-0 flex-1">
          <h3 className="line-clamp-2 text-sm font-semibold leading-snug">{product.name}</h3>
          <div className="mt-1 flex flex-wrap items-baseline gap-x-2 text-sm">
            <span className="text-base font-bold">{product.price ?? 'Price n/a'}</span>
            <span className="text-stone-500">{product.source}</span>
          </div>
          {(product.rating || product.reviews) && (
            <div className="text-xs text-stone-500">
              ★ {product.rating ?? '–'}
              {product.reviews ? ` · ${product.reviews} reviews` : ''}
            </div>
          )}
          {badges.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-1">
              {badges.map((b) => (
                <span key={b} className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-800 dark:bg-amber-950 dark:text-amber-200">
                  {b}
                  {b === 'Offer' && product.shopping_signals?.offer?.savings_percent ? ` −${product.shopping_signals.offer.savings_percent}%` : ''}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="flex-1 space-y-2 px-3 pb-3 text-sm">
        {product.why_suggested && (
          <p>
            <span className="font-medium">Why: </span>
            {product.why_suggested}
          </p>
        )}
        {product.trade_off && (
          <p className="text-stone-600 dark:text-stone-400">
            <span className="font-medium">Trade-off: </span>
            {product.trade_off}
          </p>
        )}
        {(product.requirement_checks ?? []).filter((c) => c.kind === 'user').length > 0 && (
          <ul className="flex flex-wrap gap-1 text-[11px]" aria-label="Your requirements">
            {product.requirement_checks!.filter((c) => c.kind === 'user').map((c) => (
              <li key={c.requirement} title={c.evidence_snippet ?? 'No supporting text found'}
                  className={`rounded px-1.5 py-0.5 font-medium ${CLAIM_STYLE[c.status]}`}>
                {c.status === 'unverified' ? '?' : '✓'} {c.requirement}: {c.label}
              </li>
            ))}
          </ul>
        )}
        {product.claims.length > 0 && (
          <ul className="space-y-1">
            {product.claims.map((c, i) => (
              <ClaimChip key={i} claim={c} />
            ))}
          </ul>
        )}
        {product.why_ranked && <p className="text-xs text-stone-500">Ranked: {product.why_ranked}</p>}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-stone-200 px-3 py-2 dark:border-stone-800">
        {product.link ? (
          <a
            href={product.link}
            target="_blank"
            rel="noreferrer"
            title={product.link_note}
            className="rounded-lg bg-leaf-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-leaf-700"
          >
            {product.link_type === 'merchant_product' ? 'View at seller' : 'Find on Google Shopping'}
          </a>
        ) : (
          <span className="text-xs text-stone-500">No link available</span>
        )}
        {product.immersive_product_page_token && <Offers token={product.immersive_product_page_token} />}
      </div>
    </article>
  )
}
