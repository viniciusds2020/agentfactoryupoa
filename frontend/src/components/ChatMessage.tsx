import ReactMarkdown from 'react-markdown'
import type { Message } from '../api'
import SourceCards from './SourceCards'

const PROFILE_LABELS: Record<string, string> = {
  general: 'Corporativo geral',
  legal: 'Juridico',
  hr: 'RH',
}

export default function ChatMessage({ msg, domainProfile }: { msg: Message; domainProfile?: string }) {
  const isUser = msg.role === 'user'

  return (
    <div className={`chat-row ${isUser ? 'user' : 'assistant'}`}>
      <div className={`bubble ${isUser ? 'user' : ''}`}>
        <div className="bubble-meta">
          {isUser ? 'Voce' : 'Agent Factory'}
          {!isUser && domainProfile ? <span className="bubble-profile">{PROFILE_LABELS[domainProfile] || domainProfile}</span> : null}
        </div>
        {isUser ? (
          <div className="msg-prose">{msg.content}</div>
        ) : (
          <div className="msg-prose">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        )}
        {!isUser && msg.sources?.length > 0 && <SourceCards sources={msg.sources} />}
      </div>
    </div>
  )
}
