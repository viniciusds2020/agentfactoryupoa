export default function WelcomeScreen({
  collection,
}: {
  collection: string
}) {
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
    </div>
  )
}
