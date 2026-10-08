export type ClaimStatus = 'stated_in_listing' | 'supported_by_search' | 'general_evidence' | 'conflicting_evidence' | 'unverified'

export interface RequirementCheck {
  requirement: string
  kind: 'user' | 'functional' | 'safety'
  status: ClaimStatus
  label: string
  evidence_snippet?: string | null
}

export interface Dimension {
  dimension: string
  why_relevant?: string | null
  priority: 'high' | 'medium' | 'low'
}

export interface Conclusion {
  verdict: 'clear_advantage' | 'potential_advantage' | 'no_clear_winner'
  label: string
  explanation?: string | null
  downgraded_from?: string
  downgrade_reason?: string
}

export interface Claim {
  claim: string
  status: ClaimStatus
  label: string
  validated: boolean
  evidence_snippet: string
  source_title?: string | null
  source_url?: string | null
  reason?: string
  kind?: string
  counter_snippet?: string
}

export interface ShoppingSignals {
  badges?: string[]
  offer?: { savings_percent?: number | null } | null
}

export interface Product {
  id: string
  name: string
  price: string | null
  price_value?: number | null
  source: string
  link: string | null
  link_type: string
  link_note: string
  image: string | null
  rating: number | string | null
  reviews: number | string | null
  material_type: string
  why_suggested?: string
  trade_off?: string
  claims: Claim[]
  requirement_checks?: RequirementCheck[]
  shopping_signals?: ShoppingSignals
  immersive_product_page_token?: string | null
  why_ranked?: string
  is_lowest_price?: boolean
}

export interface Material {
  type: string
  matches_request: boolean
  about: string
  impact: { material_note?: string; reusability?: string; end_of_life?: string; common_trade_off?: string }
  products: Product[]
}

export interface TraceStep {
  step: number
  query: string
  engine: string
  purpose?: string
  results_found: number
  cached?: boolean
  error?: string
  model_query?: string
  from_critique?: boolean
}

export interface AgentResult {
  query?: string
  summary: string
  product_type?: string
  functional_requirements?: string[]
  user_requirements?: string[]
  safety_requirements?: string[]
  environmental_dimensions?: Dimension[]
  focus_note?: string | null
  conclusion?: Conclusion
  unaddressed_request_terms?: string[]
  user_specified_material?: string | null
  request_sustainability?: 'eco_leaning' | 'high_impact' | 'none'
  caution_note?: string | null
  materials: Material[]
  top_picks?: Product[]
  cheapest_found?: Product | null
  no_suitable_alternative?: boolean
  availability_note?: string
  trace: TraceStep[]
  meta: Record<string, unknown> & {
    searches?: number
    evidence_searches?: number
    dropped_type_mismatch?: number
    removed_high_impact?: number
    dropped_functional_mismatch?: number
    critique?: { ran?: boolean; passed?: boolean | null; refused_removals?: unknown[] }
  }
}

export type AgentEvent =
  | { type: 'phase'; phase: string; message: string; replay?: boolean }
  | { type: 'search'; step: number; query: string; engine: string; purpose?: string }
  | { type: 'search_done'; step: number; results_found: number; cached: boolean; error?: string | null }
  | { type: 'thought'; text: string }
  | { type: 'fallback'; provider: string; errors: string[] }
  | { type: 'llm_wait'; provider: string; error: string }
  | { type: 'validate'; products: number; dropped_type_mismatch: number; removed_high_impact: number; dropped_unknown_ids: number; removed_requirement_mismatch?: number; claims: Record<string, number> }
  | { type: 'critique'; ran: boolean; passed: boolean | null; issues: unknown[]; removed: number; refused?: number; recheck?: boolean }
  | { type: 'conclusion'; verdict: string; label: string }
  | { type: 'result'; run_id: string; raw: AgentResult; view: AgentResult }
  | { type: 'error'; message: string }

export interface Offer {
  merchant: string
  logo?: string
  title?: string
  link?: string
  price?: string
  extracted_price?: number
  original_price?: string
  total?: string
  rating?: number
  reviews?: number
  tag?: string
}

export interface DemoSummary {
  slug: string
  query: string
  summary?: string
  products: number
}
