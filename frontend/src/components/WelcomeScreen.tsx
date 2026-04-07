export default function WelcomeScreen({
  collection,
  mode,
  onSuggestionClick,
}: {
  collection: string
  mode?: 'general' | 'legal' | 'tabular'
  onSuggestionClick?: (question: string) => void
}) {
  const tabularSuggestions = [
    'Qual a cobertura do procedimento 10049?',
    'Quais procedimentos sao de emergencia?',
    'Quais procedimentos tem prazo maior que 5 dias?',
    'Relatorio de prazos',
    'Alertas de SLA',
  ]
  return (
    <div className="empty-state">
      <div className="empty-glow" />
      <div className="empty-logo">
        <span className="logo-nexus">AGENT</span>
        <span className="logo-ai">FACTORY</span>
      </div>
      <div className="empty-subtitle">
        Sua consulta esta conectada a base <strong>{collection}</strong>. O assistente vai responder
        com base nos documentos disponiveis.
      </div>

      <div className="empty-domain-note">
        Esta conversa usa o modo configurado na sua base de conhecimento.
      </div>

      <div className="empty-upload glass-panel">
        <div className="empty-upload-title">Converse com sua base ja indexada</div>
        <div className="empty-upload-subtitle">
          Use a aba <strong>Ingestao</strong> para enviar arquivos e acompanhar as camadas Bronze, Prata e
          Ouro. Depois, volte aqui para consultar.
        </div>
      </div>

      {mode === 'tabular' && (
        <div className="empty-upload glass-panel" style={{ marginTop: 16 }}>
          <div className="empty-upload-title">Sugestoes para catalogos e tabelas</div>
          <div className="empty-upload-subtitle">
            Clique em uma pergunta para testar a base rapidamente.
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 12 }}>
            {tabularSuggestions.map((question) => (
              <button
                key={question}
                type="button"
                className="top-badge"
                style={{ cursor: 'pointer' }}
                onClick={() => onSuggestionClick?.(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
