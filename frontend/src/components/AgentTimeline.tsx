import type { AgentEvent } from '../types'

const PHASES: Record<string, string> = {
  identify: 'Identify product',
  gather: 'Search & gather',
  finalize: 'Assemble answer',
  validate: 'Verify in code',
  critique: 'Self-critique',
  critique_search: 'Follow-up search',
  done: 'Done',
}

interface Row {
  key: string
  icon: string
  title: string
  detail?: string
  tone: 'neutral' | 'good' | 'warn' | 'bad' | 'running'
}

function toRows(events: AgentEvent[]): Row[] {
  const rows: Row[] = []
  const searchRow = new Map<number, Row>()
  const waitRow = new Map<string, { row: Row; count: number }>()
  events.forEach((e, i) => {
    switch (e.type) {
      case 'phase':
        rows.push({ key: `p${i}`, icon: '◆', title: PHASES[e.phase] ?? e.phase, detail: e.message, tone: e.phase === 'done' ? 'good' : 'neutral' })
        break
      case 'thought':
        rows.push({ key: `t${i}`, icon: '💭', title: 'Agent reasoning', detail: e.text, tone: 'neutral' })
        break
      case 'search': {
        const row: Row = {
          key: `s${e.step}-${i}`,
          icon: e.engine === 'shopping' ? '🛒' : '🌐',
          title: `“${e.query}”`,
          detail: `${e.engine} · ${e.purpose ?? 'discovery'} · searching…`,
          tone: 'running',
        }
        searchRow.set(e.step, row)
        rows.push(row)
        break
      }
      case 'search_done': {
        const row = searchRow.get(e.step)
        if (row) {
          row.tone = e.error ? 'bad' : 'good'
          row.detail = row.detail!.replace(
            'searching…',
            e.error ? `failed: ${e.error}` : `${e.results_found} results${e.cached ? ' · cached' : ' · live SerpApi'}`,
          )
        }
        break
      }
      case 'llm_wait': {
        // One row per provider, updated in place, so a long outage doesn't flood the timeline.
        const existing = waitRow.get(e.provider)
        if (existing) {
          existing.count += 1
          existing.row.detail = `Retrying or switching AI provider… (${existing.count} attempts)`
        } else {
          const row: Row = { key: `w${e.provider}`, icon: '⏳', title: `${e.provider} is busy or rate-limited`, detail: 'Retrying or switching AI provider…', tone: 'warn' }
          waitRow.set(e.provider, { row, count: 1 })
          rows.push(row)
        }
        break
      }
      case 'fallback':
        rows.push({ key: `f${i}`, icon: '↪', title: `Switched LLM to ${e.provider}`, detail: 'Primary provider was busy; the fallback kept the run going', tone: 'warn' })
        break
      case 'validate': {
        const c = e.claims ?? {}
        const parts = [
          `${e.products} products kept`,
          e.dropped_type_mismatch ? `${e.dropped_type_mismatch} wrong-type dropped` : null,
          e.removed_high_impact ? `${e.removed_high_impact} high-impact removed` : null,
          e.dropped_unknown_ids ? `${e.dropped_unknown_ids} invented ids rejected` : null,
          e.removed_requirement_mismatch ? `${e.removed_requirement_mismatch} failed your requirements` : null,
          `claims: ${c.stated_in_listing ?? 0} seller · ${c.supported_by_search ?? 0} evidence · ${c.unverified ?? 0} unverified`,
        ].filter(Boolean)
        rows.push({ key: `v${i}`, icon: '🛡', title: 'Code guardrails applied', detail: parts.join(' · '), tone: 'good' })
        break
      }
      case 'critique':
        rows.push({
          key: `c${i}`,
          icon: '🔍',
          title: e.recheck ? 'Re-check after follow-up' : 'Self-critique',
          detail: !e.ran
            ? 'Critique unavailable; answer shown without it'
            : `${e.passed ? 'Passed' : `${e.issues.length} issue(s) noted`}${e.removed ? ` · ${e.removed} unsuitable removed` : ''}${e.refused ? ` · ${e.refused} unjustified removal(s) refused by code` : ''}`,
          tone: !e.ran ? 'warn' : e.passed ? 'good' : 'warn',
        })
        break
      case 'conclusion':
        rows.push({ key: `k${i}`, icon: '⚖', title: 'Conclusion', detail: e.label, tone: e.verdict === 'no_clear_winner' ? 'neutral' : 'good' })
        break
      case 'error':
        rows.push({ key: `e${i}`, icon: '✕', title: 'Error', detail: e.message, tone: 'bad' })
        break
    }
  })
  return rows
}

const TONE: Record<Row['tone'], string> = {
  neutral: 'border-stone-300 dark:border-stone-700',
  good: 'border-leaf-500',
  warn: 'border-amber-500',
  bad: 'border-red-500',
  running: 'border-sky-500 animate-pulse',
}

export function AgentTimeline({ events, running }: { events: AgentEvent[]; running: boolean }) {
  const rows = toRows(events)
  if (!rows.length) return null
  return (
    <section aria-label="Agent progress" className="rounded-2xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-stone-500">Agent at work</h2>
        {running && <span className="text-xs text-sky-600 dark:text-sky-400">live</span>}
      </div>
      <ol className="space-y-2">
        {rows.map((r) => (
          <li key={r.key} className={`border-l-2 pl-3 ${TONE[r.tone]}`}>
            <div className="text-sm font-medium break-words">
              <span className="mr-1.5" aria-hidden>{r.icon}</span>
              {r.title}
            </div>
            {r.detail && <div className="text-xs text-stone-500 dark:text-stone-400 break-words">{r.detail}</div>}
          </li>
        ))}
      </ol>
    </section>
  )
}
