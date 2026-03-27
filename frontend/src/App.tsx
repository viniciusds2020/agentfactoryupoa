import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowDown, Download, Loader2, Menu, Paperclip, RefreshCw, Send, Trash2, Upload, X } from 'lucide-react'
import { api } from './api'
import type {
  CollectionInfo,
  Conversation,
  DocumentArtifacts,
  DocumentRecord,
  IngestionJob,
  Message,
} from './api'
import Sidebar from './components/Sidebar'
import ChatMessage from './components/ChatMessage'
import WelcomeScreen from './components/WelcomeScreen'

const DEFAULT_EMBED = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024 // 2 GB
const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.txt', '.md', '.xlsx', '.xls', '.csv']
const MODE_STORAGE_KEY = 'kb-mode-locks-v1'

type ViewTab = 'chat' | 'ingestion'
type BaseMode = 'general' | 'legal' | 'tabular'

type BaseModeState = {
  mode: BaseMode
  locked: boolean
  locked_at?: string
}

const MODE_LABELS: Record<BaseMode, string> = {
  general: 'Conversa Geral',
  legal: 'Juridico/Contratos',
  tabular: 'Planilhas/Tabelas',
}

const MODE_HELP: Record<BaseMode, string> = {
  general: 'Bom para documentos corporativos, politicas e comunicados.',
  legal: 'Prioriza estrutura juridica para contratos, estatutos e regras formais.',
  tabular: 'Foco em tabelas, planilhas e dados estruturados.',
}

const MODE_TO_INGEST_PROFILE: Record<BaseMode, string> = {
  general: 'general',
  legal: 'legal',
  tabular: 'tabular',
}

const MODE_TO_CHAT_PROFILE: Record<BaseMode, string> = {
  general: 'general',
  legal: 'legal',
  tabular: 'general',
}

function loadModeStateMap(): Record<string, BaseModeState> {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(MODE_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, BaseModeState>
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function saveModeStateMap(map: Record<string, BaseModeState>) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(MODE_STORAGE_KEY, JSON.stringify(map))
}

function stageLabel(status: string) {
  if (status === 'queued') return 'Fila (Bronze)'
  if (status === 'bronze_received') return 'Bronze recebido'
  if (status === 'silver_processing') return 'Prata processando'
  if (status === 'processing') return 'Processando'
  if (status === 'indexed') return 'Ouro indexado'
  if (status === 'failed') return 'Falhou'
  return status
}

function stageProgress(status: string) {
  if (status === 'queued') return 5
  if (status === 'bronze_received') return 20
  if (status === 'silver_processing') return 45
  if (status === 'silver_extracted') return 65
  if (status === 'gold_indexing') return 85
  if (status === 'processing') return 80
  if (status === 'indexed') return 100
  if (status === 'failed') return 100
  return 0
}

export default function App() {
  const [collections, setCollections] = useState<CollectionInfo[]>([])
  const [collection, setCollection] = useState('')
  const [embedModel, setEmbedModel] = useState(DEFAULT_EMBED)
  const [baseMode, setBaseMode] = useState<BaseMode>('general')
  const [modeStateMap, setModeStateMap] = useState<Record<string, BaseModeState>>(loadModeStateMap)
  const [activeConv, setActiveConv] = useState<Conversation | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sidebarKey, setSidebarKey] = useState(0)
  const [showScrollBtn, setShowScrollBtn] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [activeTab, setActiveTab] = useState<ViewTab>('chat')

  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState<'ok' | 'err' | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [jobs, setJobs] = useState<IngestionJob[]>([])
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [jobsSupported, setJobsSupported] = useState(true)
  const [deletingDocId, setDeletingDocId] = useState<string | null>(null)
  const [deletingCollection, setDeletingCollection] = useState(false)
  const [artifactPreview, setArtifactPreview] = useState<DocumentArtifacts | null>(null)
  const [artifactLoadingId, setArtifactLoadingId] = useState<string | null>(null)
  const [artifactError, setArtifactError] = useState<string | null>(null)
  const [artifactTab, setArtifactTab] = useState<'markdown' | 'json'>('markdown')
  const [artifactSourceDoc, setArtifactSourceDoc] = useState<DocumentRecord | null>(null)

  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const toBottom = (smooth = true) =>
    bottomRef.current?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' })

  const refreshCollections = useCallback(async () => {
    try {
      try {
        const workspaces = await api.listWorkspaces()
        const ws = workspaces.find((w) => w.is_default) || workspaces[0]
        if (ws) api.setWorkspaceApiKey(ws.api_key)
      } catch {
        // simple mode fallback
      }

      const cols = await api.listCollections()
      setCollections(cols)
      return cols
    } catch {
      setCollections([])
      return []
    }
  }, [])

  const refreshIngestionData = useCallback(async () => {
    try {
      const docs = await api.listDocuments(collection || undefined)
      setDocuments(docs)
    } catch {
      setDocuments([])
    }

    if (!jobsSupported) return

    try {
      const data = await api.listIngestionJobs(40)
      setJobs(data)
      setJobsSupported(true)
    } catch (err) {
      const msg = err instanceof Error ? err.message : ''
      if (msg.includes('404') || msg.includes('405')) {
        setJobsSupported(false)
        setJobs([])
      }
    }
  }, [collection, jobsSupported])

  const currentModeState = collection ? modeStateMap[collection] : undefined
  const isModeLocked = !!currentModeState?.locked
  const effectiveIngestProfile = MODE_TO_INGEST_PROFILE[baseMode]
  const effectiveChatProfile = MODE_TO_CHAT_PROFILE[baseMode]

  const setModeStateForCollection = useCallback(
    (targetCollection: string, next: BaseModeState) => {
      setModeStateMap((prev) => {
        const updated = { ...prev, [targetCollection]: next }
        saveModeStateMap(updated)
        return updated
      })
    },
    [],
  )

  useEffect(() => {
    const boot = async () => {
      const cols = await refreshCollections()
      if (cols.length > 0 && !collection) {
        setCollection(cols[0].collection)
        setEmbedModel(cols[0].embedding_model)
      }
      await refreshIngestionData()
    }
    void boot()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    void refreshIngestionData()
  }, [collection, refreshIngestionData])

  useEffect(() => {
    if (!collection) return
    const saved = modeStateMap[collection]
    setBaseMode(saved?.mode || 'general')
  }, [collection, modeStateMap])

  useEffect(() => {
    if (!collection) return
    if (documents.some((doc) => doc.collection === collection && doc.status === 'indexed') && !isModeLocked) {
      setModeStateForCollection(collection, {
        mode: baseMode,
        locked: true,
        locked_at: new Date().toISOString(),
      })
    }
  }, [collection, documents, isModeLocked, baseMode, setModeStateForCollection])

  useEffect(() => {
    if (activeTab === 'ingestion') {
      void refreshIngestionData()
    }
  }, [activeTab, refreshIngestionData])

  const hasActiveJobs =
    jobsSupported &&
    jobs.some((job) =>
      ['queued', 'bronze_received', 'silver_processing', 'silver_extracted', 'gold_indexing', 'processing'].includes(
        job.status,
      ),
    )

  const visibleJobs = [...jobs]
    .sort((a, b) => {
      const aTs = Date.parse(a.finished_at || a.started_at || '') || 0
      const bTs = Date.parse(b.finished_at || b.started_at || '') || 0
      return bTs - aTs
    })
    .slice(0, 2)

  useEffect(() => {
    if (activeTab !== 'ingestion') return
    const delayMs = hasActiveJobs ? 1200 : 9000
    const timer = window.setTimeout(() => {
      void refreshIngestionData()
    }, delayMs)
    return () => window.clearTimeout(timer)
  }, [activeTab, hasActiveJobs, refreshIngestionData, jobs])

  const onCollectionChange = useCallback(
    (name: string) => {
      const isNewCollection = !collections.some((c) => c.collection === name)
      setCollection(name)
      setDocuments([])
      if (isNewCollection) {
        setBaseMode('general')
        setModeStateForCollection(name, {
          mode: 'general',
          locked: false,
        })
      }
      const match = collections.find((c) => c.collection === name)
      if (match) {
        setEmbedModel(match.embedding_model)
      }
      setActiveConv(null)
      setMessages([])
      setError(null)
      setInput('')
    },
    [collections, setModeStateForCollection],
  )

  const onModeChange = (mode: BaseMode) => {
    if (!collection || isModeLocked) return
    setBaseMode(mode)
    setModeStateForCollection(collection, {
      mode,
      locked: false,
    })
  }

  useEffect(() => {
    toBottom()
  }, [messages, loading])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const fn = () => setShowScrollBtn(el.scrollHeight - el.scrollTop - el.clientHeight > 160)
    el.addEventListener('scroll', fn)
    return () => el.removeEventListener('scroll', fn)
  }, [])

  const clearSelectedFile = () => {
    setFile(null)
    setUploadMsg(null)
    setUploadError(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  const validateFile = (selected: File) => {
    const normalizedName = selected.name.toLowerCase()
    const hasAllowedExtension = ALLOWED_EXTENSIONS.some((ext) => normalizedName.endsWith(ext))
    if (!hasAllowedExtension) return 'Formato nao suportado. Envie PDF, DOCX, PPTX, XLSX, CSV, TXT ou MD.'
    if (selected.size > MAX_UPLOAD_BYTES) return 'Arquivo excede o limite de 2 GB.'
    return null
  }

  const handleUpload = async () => {
    if (!file) return
    const validationError = validateFile(file)
    if (validationError) {
      setUploadMsg('err')
      setUploadError(validationError)
      return
    }
    setUploading(true)
    setUploadMsg(null)
    setUploadError(null)
    try {
      let asyncAccepted = false
      try {
        const job = await api.ingestDocumentAsync(
          collection || 'documentos',
          embedModel,
          file,
          effectiveIngestProfile,
        )
        asyncAccepted = true
        setUploadMsg('ok')
        setUploadError(`Arquivo recebido. Job ${job.id.slice(0, 8)} em andamento.`)
      } catch (err) {
        const msg = err instanceof Error ? err.message : ''
        const shouldFallback = msg.includes('404') || msg.includes('405')
        if (!shouldFallback) throw err
      }

      if (!asyncAccepted) {
        await api.ingestDocument(collection || 'documentos', embedModel, file, effectiveIngestProfile)
        setUploadMsg('ok')
        setUploadError('Documento processado e indexado com sucesso.')
      }

      if (collection) {
        setModeStateForCollection(collection, {
          mode: baseMode,
          locked: true,
          locked_at: new Date().toISOString(),
        })
      }

      setFile(null)
      if (fileRef.current) fileRef.current.value = ''
      await refreshCollections()
      await refreshIngestionData()
      setSidebarKey((k) => k + 1)
    } catch (err) {
      setUploadMsg('err')
      setUploadError(err instanceof Error ? err.message : 'Erro ao enviar arquivo.')
    } finally {
      setUploading(false)
    }
  }

  const deleteDocument = async (doc: DocumentRecord) => {
    setDeletingDocId(doc.id)
    try {
      await api.deleteDocument(doc.doc_id, doc.collection, doc.embedding_model)
      await refreshIngestionData()
      await refreshCollections()
      setSidebarKey((k) => k + 1)
    } finally {
      setDeletingDocId(null)
    }
  }

  const deleteCurrentCollection = async () => {
    if (!collection) return
    if (documents.length === 0) {
      window.alert('Esta base nao possui documentos para apagar.')
      return
    }
    const ok = window.confirm(
      `Deseja apagar TODOS os ${documents.length} documentos da base "${collection}"? Esta acao nao pode ser desfeita.`,
    )
    if (!ok) return

    setDeletingCollection(true)
    try {
      for (const doc of documents) {
        await api.deleteDocument(doc.doc_id, doc.collection, doc.embedding_model)
      }
      await refreshIngestionData()
      await refreshCollections()
      setSidebarKey((k) => k + 1)
      setUploadMsg('ok')
      setUploadError(`Base "${collection}" limpa com sucesso.`)
    } catch (err) {
      setUploadMsg('err')
      setUploadError(err instanceof Error ? err.message : 'Falha ao apagar base.')
    } finally {
      setDeletingCollection(false)
    }
  }

  const openArtifacts = async (doc: DocumentRecord) => {
    setArtifactLoadingId(doc.id)
    setArtifactError(null)
    try {
      const artifacts = await api.getDocumentArtifacts(doc.doc_id, doc.collection, doc.embedding_model)
      setArtifactPreview(artifacts)
      setArtifactTab(artifacts.markdown_preview ? 'markdown' : 'json')
      setArtifactSourceDoc(doc)
    } catch (err) {
      setArtifactError(err instanceof Error ? err.message : 'Falha ao carregar artefatos.')
      setArtifactPreview(null)
      setArtifactSourceDoc(null)
    } finally {
      setArtifactLoadingId(null)
    }
  }

  const downloadArtifact = async (kind: 'markdown' | 'json') => {
    if (!artifactPreview || !artifactSourceDoc) return
    const blob = await api.downloadArtifact(
      artifactSourceDoc.doc_id,
      artifactSourceDoc.collection,
      kind,
      artifactSourceDoc.embedding_model,
    )
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    const ext = kind === 'markdown' ? 'md' : 'json'
    a.href = url
    a.download = `${artifactPreview.doc_id}.${ext}`
    document.body.appendChild(a)
    a.click()
    a.remove()
    window.URL.revokeObjectURL(url)
  }

  const selectConv = async (conv: Conversation) => {
    setActiveConv(conv)
    setCollection(conv.collection)
    setEmbedModel(conv.embedding_model || DEFAULT_EMBED)
    setError(null)
    try {
      const msgs = await api.getMessages(conv.id)
      setMessages(msgs)
      setTimeout(() => toBottom(false), 50)
    } catch {
      setMessages([])
      setError('Nao foi possivel carregar a conversa.')
    }
  }

  const newConv = () => {
    setActiveConv(null)
    setMessages([])
    setError(null)
    setInput('')
    setActiveTab('chat')
  }

  const send = async (text: string) => {
    const q = text.trim()
    if (!q || loading) return
    setError(null)

    const userMsg: Message = { role: 'user', content: q, sources: [] }
    const optimistic = [...messages, userMsg]
    setMessages(optimistic)
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    setLoading(true)
    try {
      const effectiveCollection = collection || 'documentos'
      const history = messages.slice(-10).map((m) => ({ role: m.role, content: m.content }))

      let streamSources: Message['sources'] = []
      let streamConvId = ''
      let tokenBuffer = ''
      const assistantIdx = optimistic.length

      await api.sendMessageStream(
        effectiveCollection,
        embedModel,
        effectiveChatProfile,
        q,
        history,
        activeConv?.id,
        (sources, conversationId) => {
          streamSources = sources
          streamConvId = conversationId
          setMessages((prev) => [...prev, { role: 'assistant', content: '', sources }])
          setLoading(false)
        },
        (token) => {
          tokenBuffer += token
          const current = tokenBuffer
          setMessages((prev) => {
            const updated = [...prev]
            if (updated[assistantIdx]) {
              updated[assistantIdx] = {
                ...updated[assistantIdx],
                content: current,
                sources: streamSources,
              }
            }
            return updated
          })
        },
        () => {
          const finalContent = tokenBuffer
          setMessages((prev) => {
            const updated = [...prev]
            if (updated[assistantIdx]) {
              updated[assistantIdx] = {
                ...updated[assistantIdx],
                content: finalContent,
                sources: streamSources,
              }
            }
            return updated
          })
        },
      )

      if (!activeConv && streamConvId) {
        const list = await api.listConversations()
        const found = list.find((c) => c.id === streamConvId)
        if (found) setActiveConv(found)
      }
      setSidebarKey((k) => k + 1)
    } catch (err) {
      setError(
        err instanceof Error && err.message
          ? err.message
          : 'Nao foi possivel gerar uma resposta. Verifique se a base possui documentos.',
      )
      setMessages(optimistic)
    } finally {
      setLoading(false)
    }
  }

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void send(input)
    }
  }

  const onInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`
  }

  const hasMessages = messages.length > 0 || loading

  return (
    <div className="app">
      <Sidebar
        key={`${sidebarKey}-sidebar`}
        activeConvId={activeConv?.id ?? null}
        collection={collection}
        collections={collections}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((s) => !s)}
        onSelectConv={selectConv}
        onNewConv={newConv}
        onCollectionChange={onCollectionChange}
      />

      <main className="main">
        {!sidebarOpen && (
          <button
            className="sidebar-open-btn"
            onClick={() => setSidebarOpen(true)}
            aria-label="Abrir sidebar"
          >
            <Menu size={18} />
          </button>
        )}

        <div className="chat-layout">
          <div className="topbar">
            <div>
              <div className="topbar-title">Agent Factory</div>
            </div>
            <div className="topbar-badges">
              {collection && <span className="top-badge">base: {collection}</span>}
              <span className="top-badge">modo: {MODE_LABELS[baseMode]}{isModeLocked ? ' (fixo)' : ''}</span>
            </div>
          </div>

          <div className="view-tabs-wrap">
            <div className="view-tabs">
              <button
                className={`view-tab ${activeTab === 'chat' ? 'active' : ''}`}
                onClick={() => setActiveTab('chat')}
              >
                Chat
              </button>
              <button
                className={`view-tab ${activeTab === 'ingestion' ? 'active' : ''}`}
                onClick={() => setActiveTab('ingestion')}
              >
                Ingestao
              </button>
            </div>
          </div>

          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.pptx,.txt,.md,.xlsx,.xls,.csv"
            className="hidden"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null)
              setUploadMsg(null)
              setUploadError(null)
            }}
          />

          {activeTab === 'chat' ? (
            <>
              <div ref={scrollRef} className="messages-scroll">
                {error && <div className="error-banner">{error}</div>}

                <div className="messages-inner">
                  {!hasMessages ? (
                    <WelcomeScreen collection={collection || 'documentos'} />
                  ) : (
                    <>
                      {messages.map((msg, i) => (
                        <ChatMessage key={i} msg={msg} domainProfile={effectiveChatProfile} />
                      ))}

                      {loading && (
                        <div className="chat-row assistant">
                          <div className="bubble">
                            <div className="bubble-meta">Agent Factory</div>
                            <div className="typing" aria-label="Gerando resposta">
                              <span />
                              <span />
                              <span />
                            </div>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                  <div ref={bottomRef} />
                </div>
              </div>

              <div className="input-shell">
                <div className="input-wrapper">
                  <textarea
                    ref={textareaRef}
                    rows={1}
                    value={input}
                    onChange={onInput}
                    onKeyDown={onKey}
                    placeholder={
                      collection
                        ? `Pergunte sobre "${collection}"...`
                        : 'Abra a aba de ingestao para enviar documentos...'
                    }
                    disabled={loading}
                    className="chat-input"
                  />
                  <button
                    className="attach-btn"
                    onClick={() => {
                      setActiveTab('ingestion')
                      fileRef.current?.click()
                    }}
                    aria-label="Ir para ingestao"
                    title="Ir para ingestao"
                  >
                    <Paperclip size={16} />
                  </button>
                  <button
                    className="send-btn"
                    onClick={() => void send(input)}
                    disabled={!input.trim() || loading}
                    aria-label="Enviar mensagem"
                  >
                    {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                  </button>
                </div>
                <div className="helper-line">Enter para enviar. Shift+Enter para nova linha.</div>
              </div>

              {showScrollBtn && (
                <button
                  className="sidebar-open-btn"
                  style={{ top: 'auto', bottom: 92, left: 'auto', right: 14 }}
                  onClick={() => toBottom()}
                  aria-label="Ir para o fim"
                >
                  <ArrowDown size={16} />
                </button>
              )}
            </>
          ) : (
            <div className="messages-scroll">
              <div className="messages-inner monitoring-stack">
                <section className="ops-panel glass-panel ops-card ops-card-pipeline">
                  <div className="ops-header">
                    <div>
                      <div className="ops-kicker">Pipeline</div>
                      <div className="ops-title">Upload e Ingestao</div>
                    </div>
                    <button className="ops-refresh" onClick={() => void refreshIngestionData()}>
                      <RefreshCw size={14} />
                      atualizar
                    </button>
                  </div>

                  <div className="ops-grid">
                    <div className="ops-section">
                      <div className="ops-section-head">
                        <span>
                          <Upload size={14} />
                          Novo arquivo
                        </span>
                      </div>
                      <div className="ops-list" style={{ marginBottom: 12 }}>
                        <div className="ops-item">
                          <div className="ops-item-main">
                            <div className="ops-item-title">Modo da Base</div>
                            <div className="ops-item-meta">
                              Escolha como esta base sera processada. Depois da primeira ingestao, fica fixo.
                            </div>
                            <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
                              {(['general', 'legal', 'tabular'] as BaseMode[]).map((mode) => (
                                <label
                                  key={mode}
                                  style={{ display: 'flex', alignItems: 'flex-start', gap: 8, cursor: isModeLocked ? 'not-allowed' : 'pointer' }}
                                >
                                  <input
                                    type="radio"
                                    name="base-mode"
                                    checked={baseMode === mode}
                                    onChange={() => onModeChange(mode)}
                                    disabled={isModeLocked}
                                  />
                                  <span>
                                    <strong>{MODE_LABELS[mode]}</strong>
                                    <br />
                                    <span className="ops-item-meta">{MODE_HELP[mode]}</span>
                                  </span>
                                </label>
                              ))}
                            </div>
                            {isModeLocked && (
                              <div className="ops-item-meta" style={{ marginTop: 8 }}>
                                Modo fixo desta base. Para outro modo, crie uma nova base.
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                      {!file ? (
                        <button className="empty-upload-btn" onClick={() => fileRef.current?.click()}>
                          <Paperclip size={14} />
                          escolher arquivo
                        </button>
                      ) : (
                        <div className="empty-upload-selected">
                          <div className="empty-upload-file">
                            <div className="empty-upload-file-label">Arquivo escolhido</div>
                            <div className="empty-upload-file-name">{file.name}</div>
                          </div>
                          <div className="empty-upload-actions">
                            <button
                              className="empty-upload-btn"
                              onClick={() => void handleUpload()}
                              disabled={uploading}
                            >
                              {uploading ? 'enviando...' : 'iniciar ingestao'}
                            </button>
                            <button className="empty-upload-remove" onClick={clearSelectedFile}>
                              <X size={14} />
                            </button>
                          </div>
                        </div>
                      )}
                      {uploadMsg && (
                        <div className={`empty-upload-status ${uploadMsg === 'err' ? 'error' : ''}`}>
                          {uploadMsg === 'ok'
                            ? uploadError || 'Ingestao iniciada.'
                            : uploadError || 'Falha no upload.'}
                        </div>
                      )}
                    </div>

                    <div className="ops-section">
                      <div className="ops-section-head">
                        <span>Camadas</span>
                      </div>
                      <div className="ops-list">
                        <div className="ops-item queued">
                          <div className="ops-item-main">
                            <div className="ops-item-title">Bronze</div>
                            <div className="ops-item-meta">Recebimento do arquivo bruto e metadados.</div>
                          </div>
                        </div>
                        <div className="ops-item processing">
                          <div className="ops-item-main">
                            <div className="ops-item-title">Prata</div>
                            <div className="ops-item-meta">Extracao e normalizacao para markdown estruturado.</div>
                          </div>
                        </div>
                        <div className="ops-item">
                          <div className="ops-item-main">
                            <div className="ops-item-title">Ouro</div>
                            <div className="ops-item-meta">Chunking, embeddings e indexacao vetorial.</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>

                <section className="ops-panel glass-panel ops-card ops-card-jobs">
                  <div className="ops-header">
                    <div>
                      <div className="ops-kicker">Execucao</div>
                      <div className="ops-title">Jobs de Ingestao</div>
                    </div>
                  </div>
                  {jobsSupported ? (
                    <div className="ops-list ops-list-compact">
                      {visibleJobs.length === 0 ? (
                        <div className="ops-empty">Nenhum job encontrado.</div>
                      ) : (
                        visibleJobs.map((job) => (
                          <div key={job.id} className={`ops-item ${job.status}`}>
                            <div className="ops-item-main">
                              <div className="ops-item-title">{job.filename || job.doc_id}</div>
                              <div className="ops-item-meta">
                                {stageLabel(job.status)} • chunks: {job.chunks_indexed}
                              </div>
                              <div className="ops-item-meta">
                                inicio: {job.started_at || '-'} • fim: {job.finished_at || '-'}
                              </div>
                              <div className="upload-progress-wrap">
                                <div className="upload-progress-bar">
                                  <div
                                    className="upload-progress-fill"
                                    style={{ width: `${job.progress_pct ?? stageProgress(job.status)}%` }}
                                  />
                                </div>
                                <div className="upload-progress-text">
                                  {job.progress_pct ?? stageProgress(job.status)}% • etapa {job.stage || 'n/a'}
                                </div>
                              </div>
                              {job.error && <div className="ops-item-error">{job.error}</div>}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  ) : (
                    <div className="ops-empty">
                      Jobs assinc disponíveis apenas quando a rota enterprise esta habilitada.
                    </div>
                  )}
                </section>

                <section className="ops-panel glass-panel ops-card ops-card-docs">
                  <div className="ops-header">
                    <div>
                      <div className="ops-kicker">Inventario</div>
                      <div className="ops-title">Documentos na Base</div>
                    </div>
                    <button
                      className="ops-refresh"
                      onClick={() => void deleteCurrentCollection()}
                      disabled={deletingCollection || documents.length === 0}
                      title="Apagar todos os documentos da base selecionada"
                    >
                      {deletingCollection ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                      apagar base
                    </button>
                  </div>
                  <div className="ops-list ops-list-compact">
                    {documents.length === 0 ? (
                      <div className="ops-empty">Sem documentos para a colecao selecionada.</div>
                    ) : (
                      documents.map((doc) => (
                        <div key={doc.id} className={`ops-item ${doc.status}`}>
                          <div className="ops-item-main">
                            <div className="ops-item-title">{doc.filename || doc.doc_id}</div>
                            <div className="ops-item-meta">
                              {stageLabel(doc.status)} • chunks: {doc.chunks_indexed}
                            </div>
                            <div className="upload-progress-wrap">
                              <div className="upload-progress-bar">
                                <div
                                  className="upload-progress-fill"
                                  style={{ width: `${stageProgress(doc.status)}%` }}
                                />
                              </div>
                              <div className="upload-progress-text">{stageProgress(doc.status)}%</div>
                            </div>
                            {doc.error && <div className="ops-item-error">{doc.error}</div>}
                          </div>
                          <div style={{ display: 'grid', gap: 8 }}>
                            <button
                              className="ops-delete"
                              onClick={() => void openArtifacts(doc)}
                              disabled={artifactLoadingId === doc.id}
                              aria-label={`Preview ${doc.filename || doc.doc_id}`}
                              title="Preview Prata (Markdown/JSON)"
                            >
                              {artifactLoadingId === doc.id ? <Loader2 size={14} className="animate-spin" /> : 'P'}
                            </button>
                            <button
                              className="ops-delete"
                              onClick={() => void deleteDocument(doc)}
                              disabled={deletingDocId === doc.id}
                              aria-label={`Excluir ${doc.filename || doc.doc_id}`}
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </section>

                <section className="ops-panel glass-panel ops-card ops-card-preview">
                  <div className="ops-header">
                    <div>
                      <div className="ops-kicker">Prata</div>
                      <div className="ops-title">Preview de Artefatos</div>
                    </div>
                  </div>
                  {artifactError && <div className="ops-item-error">{artifactError}</div>}
                  {!artifactPreview && !artifactError && (
                    <div className="ops-empty">
                      Selecione um documento em "Documentos na Base" para visualizar markdown/json extraido.
                    </div>
                  )}
                  {artifactPreview && (
                    <div className="ops-section">
                      <div className="ops-section-head">
                        <span>{artifactPreview.doc_id}</span>
                        <div style={{ display: 'inline-flex', gap: 8 }}>
                          <button
                            className="ops-refresh"
                            onClick={() => void downloadArtifact('markdown')}
                            disabled={!artifactPreview.markdown_preview}
                            title="Baixar markdown"
                          >
                            <Download size={14} />
                            md
                          </button>
                          <button
                            className="ops-refresh"
                            onClick={() => void downloadArtifact('json')}
                            disabled={!artifactPreview.json_preview}
                            title="Baixar json"
                          >
                            <Download size={14} />
                            json
                          </button>
                        </div>
                      </div>
                      <div className="view-tabs" style={{ marginBottom: 10 }}>
                        <button
                          className={`view-tab ${artifactTab === 'markdown' ? 'active' : ''}`}
                          onClick={() => setArtifactTab('markdown')}
                          disabled={!artifactPreview.markdown_preview}
                        >
                          Markdown
                        </button>
                        <button
                          className={`view-tab ${artifactTab === 'json' ? 'active' : ''}`}
                          onClick={() => setArtifactTab('json')}
                          disabled={!artifactPreview.json_preview}
                        >
                          JSON
                        </button>
                      </div>
                      <div className="source-card" style={{ maxHeight: 360, overflow: 'auto' }}>
                        <pre className="msg-prose" style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                          {artifactTab === 'markdown'
                            ? artifactPreview.markdown_preview || 'Sem markdown disponivel.'
                            : artifactPreview.json_preview || 'Sem JSON disponivel.'}
                        </pre>
                      </div>
                    </div>
                  )}
                </section>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
