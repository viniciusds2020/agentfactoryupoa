import { useEffect, useState } from 'react'
import { ChevronDown, MessageSquare, PanelLeftClose, Plus, Trash2 } from 'lucide-react'
import { api } from '../api'
import type { CollectionInfo, Conversation } from '../api'

interface Props {
  activeConvId: string | null
  collection: string
  collections: CollectionInfo[]
  sidebarOpen: boolean
  onToggleSidebar: () => void
  onSelectConv: (conv: Conversation) => void
  onNewConv: () => void
  onCollectionChange: (c: string) => void
}

export default function Sidebar(props: Props) {
  const {
    activeConvId,
    collection,
    collections,
    sidebarOpen,
    onToggleSidebar,
    onSelectConv,
    onNewConv,
    onCollectionChange,
  } = props

  const [convs, setConvs] = useState<Conversation[]>([])
  const [search, setSearch] = useState('')
  const [creatingNew, setCreatingNew] = useState(false)
  const [newColName, setNewColName] = useState('')
  const [helpOpen, setHelpOpen] = useState(false)

  useEffect(() => {
    const run = async () => {
      try {
        const all = await api.listConversations(search)
        setConvs(collection ? all.filter((conv) => conv.collection === collection) : all)
      } catch {
        setConvs([])
      }
    }
    void run()
  }, [search, collection])

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    await api.deleteConversation(id)
    const all = await api.listConversations(search).catch(() => [])
    setConvs(collection ? all.filter((conv) => conv.collection === collection) : all)
    if (activeConvId === id) onNewConv()
  }

  const uniqueCollections = Array.from(new Set(collections.map((c) => c.collection))).sort()

  const confirmNewCollection = () => {
    const name = newColName.trim().replace(/[^a-zA-Z0-9_-]/g, '')
    if (name) {
      onCollectionChange(name)
    }
    setNewColName('')
    setCreatingNew(false)
  }

  return (
    <aside className={`sidebar ${sidebarOpen ? '' : 'closed'}`}>
      <div className="sidebar-inner">
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <span className="brand-nexus">AGENT</span>
            <span className="brand-ai">FACTORY</span>
          </div>
          <button className="toggle-btn" onClick={onToggleSidebar} aria-label="Fechar sidebar">
            <PanelLeftClose size={16} />
          </button>
        </div>

        <button className="new-chat-btn" onClick={onNewConv}>
          <Plus size={16} />
          Nova conversa
        </button>

        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Buscar conversa"
          className="sidebar-search"
        />

        <div className="sidebar-divider">
          <span>Conversas desta base</span>
        </div>

        <div className="conversations">
          {convs.length === 0 ? (
            <div className="conversations-empty">
              {search ? 'Nenhum resultado encontrado.' : 'Nenhuma conversa ainda.'}
            </div>
          ) : (
            convs.map((conv) => {
              const active = conv.id === activeConvId
              return (
                <div key={conv.id} className={`conv-item ${active ? 'active' : ''}`}>
                  <button className="conv-select" onClick={() => onSelectConv(conv)}>
                    <MessageSquare size={14} className="conv-icon" />
                    <span className="conv-title">{conv.title}</span>
                  </button>
                  <button
                    className="conv-clear"
                    onClick={(e) => void handleDelete(e, conv.id)}
                    aria-label={`Excluir conversa ${conv.title}`}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              )
            })
          )}
        </div>

        <div className="sidebar-footer">
          <div>
            <button
              type="button"
              className="sidebar-help-toggle"
              onClick={() => setHelpOpen((open) => !open)}
              aria-expanded={helpOpen}
            >
              <span className="footer-label">Como usar</span>
              <ChevronDown
                size={13}
                style={{
                  transform: helpOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                }}
              />
            </button>
            {helpOpen && (
              <div className="sidebar-todo">
                <div className="sidebar-todo-item">
                  <span className="sidebar-todo-dot" />
                  <span>Use a aba de ingestao para enviar e indexar documentos.</span>
                </div>
                <div className="sidebar-todo-item">
                  <span className="sidebar-todo-dot" />
                  <span>Escolha a base de documentos que deseja consultar.</span>
                </div>
                <div className="sidebar-todo-item">
                  <span className="sidebar-todo-dot" />
                  <span>Faca perguntas e confira os trechos usados em cada resposta.</span>
                </div>
              </div>
            )}
          </div>

          <div>
            <div className="sidebar-row-head">
              <div className="footer-label">Base de documentos</div>
              <button
                type="button"
                onClick={() => {
                  setCreatingNew((v) => !v)
                  setNewColName('')
                }}
                title="Criar nova base"
                className="sidebar-inline-action"
              >
                <Plus size={13} />
                nova
              </button>
            </div>
            {creatingNew ? (
              <input
                autoFocus
                value={newColName}
                onChange={(e) => setNewColName(e.target.value)}
                onBlur={confirmNewCollection}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') confirmNewCollection()
                  if (e.key === 'Escape') {
                    setCreatingNew(false)
                    setNewColName('')
                  }
                }}
                placeholder="Nome da nova base"
                className="sidebar-field"
              />
            ) : (
              <select
                value={collection}
                onChange={(e) => onCollectionChange(e.target.value)}
                className="sidebar-select"
              >
                {uniqueCollections.length === 0 && (
                  <option value={collection || 'documentos'}>
                    {collection || 'documentos'}
                  </option>
                )}
                {uniqueCollections.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            )}
          </div>

        </div>
      </div>
    </aside>
  )
}
