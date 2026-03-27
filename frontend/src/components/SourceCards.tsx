import { useState } from 'react'
import { ChevronDown, FileText } from 'lucide-react'
import type { Source } from '../api'

export default function SourceCards({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  if (!sources.length) return null

  return (
    <div>
      <button className="source-toggle" onClick={() => setOpen((o) => !o)}>
        {sources.length} fonte{sources.length > 1 ? 's' : ''} consultada{sources.length > 1 ? 's' : ''}
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
            const sourceName = src.source_filename || src.doc_id || src.chunk_id.slice(0, 12)
            const metaLine = src.citation_label || (src.page_number ? `pagina ${src.page_number}` : 'trecho referenciado')
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
                <div className="source-excerpt">{src.excerpt}</div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
