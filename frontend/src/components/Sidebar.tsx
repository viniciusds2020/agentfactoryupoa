import { useEffect, useMemo, useState } from 'react'
import { ChevronDown, Database, MessageSquare, PanelLeftClose, Plus, Rows3, Trash2, Upload } from 'lucide-react'
import { api } from '../api'
import type { CollectionInfo, Conversation } from '../api'
import { MODE_LABELS } from '../appModes'
import type { BaseMode } from '../appModes'

interface Props {
  activeConvId: string | null
  activeTab: 'chat' | 'ingestion'
  baseMode: BaseMode
  collection: string
  collections: CollectionInfo[]
  documentsCount: number
  indexedDocumentsCount: number
  jobsSupported: boolean
  activeJobsCount: number
  isModeLocked: boolean
  sidebarOpen: boolean
  onToggleSidebar: () => void
  onSelectConv: (conv: Conversation) => void
  onSelectTab: (tab: 'chat' | 'ingestion') => void
  onNewConv: () => void
  onCollectionChange: (c: string) => void
}

export default function Sidebar(props: Props) {
  const {
    activeConvId,
    activeTab,
    baseMode,
    collection,
    collections,
    documentsCount,
    indexedDocumentsCount,
    jobsSupported,
    activeJobsCount,
    isModeLocked,
    sidebarOpen,
    onToggleSidebar,
    onSelectConv,
    onSelectTab,
    onNewConv,
    onCollectionChange,
  } = props

  const [convs, setConvs] = useState<Conversation[]>([])
  const [search, setSearch] = useState('')
  const [creatingNew, setCreatingNew] = useState(false)
  const [newColName, setNewColName] = useState('')
  const [processOpen, setProcessOpen] = useState(false)
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
  const processSteps = useMemo(() => {
    const hasBase = !!collection
    const hasDocs = documentsCount > 0
    const hasIndexedDocs = indexedDocumentsCount > 0

    return [
      {
        id: 'base',
        label: 'Base selecionada',
        meta: hasBase ? collection : 'escolha ou crie uma base',
        status: hasBase ? 'done' : activeTab === 'ingestion' ? 'active' : 'idle',
      },
      {
        id: 'ingest',
        label: 'Ingestao',
        meta: hasDocs ? `${documentsCount} documento(s)` : 'envie o primeiro arquivo',
        status: hasDocs ? 'done' : activeTab === 'ingestion' ? 'active' : 'idle',
      },
      {
        id: 'validate',
        label: 'Processamento',
        meta: hasIndexedDocs
          ? `${indexedDocumentsCount} indexado(s)`
          : jobsSupported && activeJobsCount > 0
            ? `${activeJobsCount} job(s) em andamento`
            : 'aguardando indexacao',
        status: hasIndexedDocs ? 'done' : activeTab === 'ingestion' && hasDocs ? 'active' : 'idle',
      },
      {
        id: 'chat',
        label: 'Conversa',
        meta: hasIndexedDocs ? 'base pronta para perguntas' : 'disponivel apos indexacao',
        status: activeTab === 'chat' && hasIndexedDocs ? 'active' : hasIndexedDocs ? 'done' : 'idle',
      },
    ] as const
  }, [activeJobsCount, activeTab, collection, documentsCount, indexedDocumentsCount, jobsSupported])

  const statusSummary = useMemo(() => {
    if (activeJobsCount > 0) return `${activeJobsCount} job(s) em andamento`
    if (indexedDocumentsCount > 0) return `${indexedDocumentsCount} documento(s) prontos`
    if (documentsCount > 0) return `${documentsCount} documento(s) carregados`
    return 'sem documentos na base'
  }, [activeJobsCount, documentsCount, indexedDocumentsCount])

  const nextAction = useMemo(() => {
    if (!collection) {
      return {
        label: 'Selecionar base',
        hint: 'Escolha ou crie uma base para iniciar o fluxo.',
        tab: 'ingestion' as const,
      }
    }
    if (documentsCount === 0) {
      return {
        label: 'Enviar primeiro arquivo',
        hint: 'A base existe, mas ainda nao possui documentos.',
        tab: 'ingestion' as const,
      }
    }
    if (activeJobsCount > 0 || indexedDocumentsCount === 0) {
      return {
        label: 'Acompanhar processamento',
        hint: 'A indexacao ainda esta em andamento.',
        tab: 'ingestion' as const,
      }
    }
    return {
      label: 'Abrir chat da base',
      hint: 'A base ja esta pronta para perguntas.',
      tab: 'chat' as const,
    }
  }, [activeJobsCount, collection, documentsCount, indexedDocumentsCount])

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

        <div className="sidebar-top-card">
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
          <div className="sidebar-base-stats">
            <div className="sidebar-base-stat">
              <Database size={12} />
              <span>{documentsCount} docs</span>
            </div>
            <div className="sidebar-base-stat">
              <Rows3 size={12} />
              <span>{convs.length} conversa(s)</span>
            </div>
          </div>
        </div>

        <div className="sidebar-process-card">
          <div className="sidebar-process-head">
            <div>
              <div className="footer-label">Fluxo da Base</div>
              <div className="sidebar-process-title">Estado atual do processo</div>
            </div>
            <div className="sidebar-process-head-actions">
              <span className="sidebar-process-chip">{MODE_LABELS[baseMode]}</span>
              <button
                type="button"
                className="sidebar-process-toggle"
                onClick={() => setProcessOpen((open) => !open)}
                aria-expanded={processOpen}
                aria-label={processOpen ? 'Recolher fluxo da base' : 'Expandir fluxo da base'}
              >
                <ChevronDown
                  size={14}
                  style={{
                    transform: processOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s',
                  }}
                />
              </button>
            </div>
          </div>

          <div className="sidebar-process-meta">
            <span>{statusSummary}</span>
            <span>{isModeLocked ? 'modo fixo' : 'modo editavel'}</span>
          </div>

          {processOpen && (
            <>
              <div className="sidebar-process-next">
                <div className="sidebar-process-next-copy">
                  <span className="footer-label">Proximo passo</span>
                  <strong>{nextAction.label}</strong>
                  <span>{nextAction.hint}</span>
                </div>
                <button
                  type="button"
                  className="sidebar-process-next-btn"
                  onClick={() => onSelectTab(nextAction.tab)}
                >
                  {nextAction.tab === 'chat' ? 'Ir para chat' : 'Ir para ingestao'}
                </button>
              </div>

              <div className="sidebar-process-actions">
                <button
                  type="button"
                  className={`sidebar-process-btn ${activeTab === 'chat' ? 'active' : ''}`}
                  onClick={() => onSelectTab('chat')}
                >
                  <MessageSquare size={13} />
                  Chat
                </button>
                <button
                  type="button"
                  className={`sidebar-process-btn ${activeTab === 'ingestion' ? 'active' : ''}`}
                  onClick={() => onSelectTab('ingestion')}
                >
                  <Upload size={13} />
                  Ingestao
                </button>
              </div>

              <div className="sidebar-process-list">
                {processSteps.map((step, index) => (
                  <div key={step.id} className={`sidebar-step ${step.status}`}>
                    <div className="sidebar-step-index">{index + 1}</div>
                    <div className="sidebar-step-body">
                      <div className="sidebar-step-label">{step.label}</div>
                      <div className="sidebar-step-meta">{step.meta}</div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Buscar conversa"
          className="sidebar-search"
        />

        <div className="sidebar-divider">
          <span>Conversas desta base</span>
          <strong>{convs.length}</strong>
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
        </div>
      </div>
    </aside>
  )
}
