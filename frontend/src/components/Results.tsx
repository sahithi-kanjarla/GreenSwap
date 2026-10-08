import type { AgentResult, Material } from '../types'
import { ProductCard } from './ProductCard'

function MaterialGroup({ material, highlight }: { material: Material; highlight?: string }) {
  const impact = material.impact ?? {}
  const facts: [string, string | undefined][] = [
    ['Material', impact.material_note],
    ['Use pattern', impact.reusability],
    ['End of life', impact.end_of_life],
    ['Trade-off', impact.common_trade_off],
  ]
  return (
    <section className="space-y-3">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-lg font-semibold capitalize">{material.type}</h3>
          {highlight && <span className="rounded-full bg-leaf-100 px-2 py-0.5 text-xs font-medium text-leaf-700 dark:bg-leaf-900 dark:text-leaf-100">{highlight}</span>}
          <span className="text-xs text-stone-500">{material.products.length} product{material.products.length === 1 ? '' : 's'}</span>
        </div>
        {material.about && <p className="text-sm text-stone-600 dark:text-stone-400">{material.about}</p>}
        <dl className="mt-2 grid gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
          {facts.filter(([, v]) => v).map(([k, v]) => (
            <div key={k}>
              <dt className="inline font-medium text-stone-700 dark:text-stone-300">{k}: </dt>
              <dd className="inline text-stone-600 dark:text-stone-400">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-1 text-[11px] text-stone-400">General context from the model, not verified per listing.</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {material.products.map((p) => <ProductCard key={p.id} product={p} />)}
      </div>
    </section>
  )
}

export function Results({ result }: { result: AgentResult }) {
  const meta = result.meta ?? {}
  const req = result.request_sustainability ?? 'none'
  const ordered = [...result.materials]

  return (
    <div className="space-y-8">
      <section className="space-y-3 rounded-2xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
        <p className="text-base">{result.summary}</p>
        {result.caution_note && (
          <div role="note" className="rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
            <strong>About {result.user_specified_material ?? 'your choice'}:</strong> {result.caution_note} We show better alternatives instead.
          </div>
        )}
        {result.conclusion && (
          <div className={`rounded-xl border p-3 text-sm ${result.conclusion.verdict === 'no_clear_winner'
            ? 'border-stone-300 bg-stone-50 dark:border-stone-700 dark:bg-stone-950'
            : 'border-leaf-500/40 bg-leaf-50 dark:border-leaf-700 dark:bg-leaf-900/40'}`}>
            <div className="font-semibold">{result.conclusion.label}</div>
            {result.conclusion.explanation && <p className="text-stone-700 dark:text-stone-300">{result.conclusion.explanation}</p>}
            {result.conclusion.downgrade_reason && (
              <p className="mt-1 text-xs text-stone-500">Toned down by GreenSwap's checks: {result.conclusion.downgrade_reason}.</p>
            )}
          </div>
        )}
        {result.focus_note && <p className="text-sm text-stone-600 dark:text-stone-400">{result.focus_note}</p>}
        <div className="flex flex-wrap gap-1.5 text-xs">
          {result.product_type && <span className="rounded-full bg-stone-100 px-2 py-1 dark:bg-stone-800">Looking for: <b>{result.product_type}</b></span>}
          {(result.user_requirements ?? []).map((r) => (
            <span key={`u-${r}`} className="rounded-full bg-leaf-100 px-2 py-1 font-medium text-leaf-700 dark:bg-leaf-900 dark:text-leaf-100">You asked: {r}</span>
          ))}
          {[...(result.functional_requirements ?? []), ...(result.safety_requirements ?? [])].map((r) => (
            <span key={r} className="rounded-full bg-stone-100 px-2 py-1 dark:bg-stone-800">✓ {r}</span>
          ))}
        </div>
        {(result.environmental_dimensions ?? []).length > 0 && (
          <div className="text-xs">
            <span className="text-stone-500">Compared on: </span>
            {result.environmental_dimensions!.map((d, i) => (
              <span key={d.dimension} title={d.why_relevant ?? ''} className="font-medium">
                {i > 0 && ' · '}{d.dimension}{d.priority !== 'high' && <span className="text-stone-400"> ({d.priority})</span>}
              </span>
            ))}
          </div>
        )}
        {(result.unaddressed_request_terms ?? []).length > 0 && (
          <p className="text-xs text-amber-700 dark:text-amber-300">
            Not explicitly checked from your request: {result.unaddressed_request_terms!.join(', ')}
          </p>
        )}
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-stone-500">
          <span>{meta.searches ?? 0} SerpApi searches</span>
          {!!meta.dropped_type_mismatch && <span>{meta.dropped_type_mismatch} wrong-type listings dropped</span>}
          {!!meta.removed_high_impact && <span>{meta.removed_high_impact} high-impact listings withheld</span>}
          {!!meta.dropped_functional_mismatch && <span>{meta.dropped_functional_mismatch} unsuitable removed by self-critique</span>}
        </div>
      </section>

      {result.top_picks && result.top_picks.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-xl font-bold">Top picks</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {result.top_picks.map((p) => <ProductCard key={`top-${p.id}`} product={p} />)}
          </div>
        </section>
      )}

      {ordered.length > 0 ? (
        <section className="space-y-8">
          <h2 className="text-xl font-bold">{req === 'high_impact' ? 'Better alternatives' : 'By alternative'}</h2>
          {ordered.map((m) => (
            <MaterialGroup key={m.type} material={m} highlight={req === 'eco_leaning' && m.matches_request ? 'What you asked for' : undefined} />
          ))}
        </section>
      ) : (
        <p className="rounded-2xl border border-dashed border-stone-300 p-6 text-center text-sm text-stone-500 dark:border-stone-700">
          No suitable alternative matched your need and budget. GreenSwap would rather show nothing than something unsuitable.
        </p>
      )}

      {result.availability_note && <p className="text-xs text-stone-500">{result.availability_note}</p>}
    </div>
  )
}
