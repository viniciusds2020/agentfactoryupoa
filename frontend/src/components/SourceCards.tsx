import { useState } from 'react'
import { ChevronDown, FileText } from 'lucide-react'
import type { Source } from '../api'

export default function SourceCards({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null
  const isTableQuery = sources.every((src) => src.source_kind === 'table_query')

  return (
    <div>
      <button className="source-toggle" onClick={() => setOpen((o) => !o)}>
        {isTableQuery
          ? `${sources.length} consulta${sources.length > 1 ? 's' : ''} analitica${sources.length > 1 ? 's' : ''} executada${sources.length > 1 ? 's' : ''}`
          : `${sources.length} fonte${sources.length > 1 ? 's' : ''} consultada${sources.length > 1 ? 's' : ''}`}
        <ChevronDown
          size={13}
          style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
        />
      </button>

      {open && (
        <div className="source-list">
          {sources.map((src, i) => {
            const maxScore = sources[0]?.score || 1
            const pct = maxScore > 0 ? (src.score / maxScore) * 100 : 0
            const score = `${Math.round(pct)}%`
            const sourceName = src.source_kind === 'table_query'
              ? 'Consulta analitica'
              : src.source_filename || src.doc_id || src.chunk_id.slice(0, 12)
            const metaLine = src.citation_label || (src.page_number ? `pagina ${src.page_number}` : 'trecho referenciado')
            const body = src.source_kind === 'table_query' ? (src.query_summary || src.excerpt) : src.excerpt
            const resultPreview = src.source_kind === 'table_query' ? (src.result_preview || '') : ''
            return (
              <div key={i} className="source-card">
                <div className="source-head">
                  <div className="source-name">
                    <FileText size={13} />
                    {sourceName}
                  </div>
                  <div className="source-score">{score}</div>
                </div>
                <div className="source-meta-line">{metaLine}</div>
                <div className="source-excerpt">{body}</div>
                {resultPreview && <pre className="source-result-preview">{resultPreview}</pre>}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
