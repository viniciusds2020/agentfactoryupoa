import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowDown, Loader2, Menu, Send } from 'lucide-react'
import { api } from './api'
import {
  inferSuggestedModeFromFile,
  loadModeStateMap,
  MODE_LABELS,
  MODE_TO_CHAT_PROFILE,
  MODE_TO_INGEST_PROFILE,
  saveModeStateMap,
} from './appModes'
import type {
  CollectionSemanticProfile,
  CollectionInfo,
  Conversation,
  DeadlineReport,
  DocumentArtifacts,
  DocumentRecord,
  IngestionJob,
  Message,
  TabularEvaluation,
} from './api'
import type { BaseMode, BaseModeState, SuggestedMode } from './appModes'
import Sidebar from './components/Sidebar'
import ChatMessage from './components/ChatMessage'
import IngestionPanel from './components/IngestionPanel'
import WelcomeScreen from './components/WelcomeScreen'

const DEFAULT_EMBED = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024 // 2 GB
const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.txt', '.md', '.xlsx', '.xls', '.csv']
type ViewTab = 'chat' | 'ingestion'
type IngestionFocusTarget = 'pipeline' | 'jobs' | 'documents' | null
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
  const [ingestionFocusTarget, setIngestionFocusTarget] = useState<IngestionFocusTarget>(null)

  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState<'ok' | 'err' | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [tableContext, setTableContext] = useState('')
  const [tableContextDirty, setTableContextDirty] = useState(false)
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
  const [semanticProfile, setSemanticProfile] = useState<CollectionSemanticProfile | null>(null)
  const [semanticLoading, setSemanticLoading] = useState(false)
  const [semanticError, setSemanticError] = useState<string | null>(null)
  const [tabularEval, setTabularEval] = useState<TabularEvaluation | null>(null)
  const [deadlineReport, setDeadlineReport] = useState<DeadlineReport | null>(null)
  const [suggestedMode, setSuggestedMode] = useState<SuggestedMode>(null)

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
    if (baseMode !== 'tabular' || tableContextDirty) return
    const existingContext = documents.find((doc) => doc.collection === collection && doc.context_hint?.trim())?.context_hint || ''
    setTableContext(existingContext)
  }, [baseMode, collection, documents, tableContextDirty])

  useEffect(() => {
    setTableContextDirty(false)
  }, [collection, baseMode])

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

  useEffect(() => {
    if (activeTab !== 'ingestion' || !collection || baseMode !== 'tabular') {
      setSemanticProfile(null)
      setSemanticError(null)
      setDeadlineReport(null)
      return
    }
    let cancelled = false
    const loadSemanticProfile = async () => {
      setSemanticLoading(true)
      setSemanticError(null)
      try {
        const [profile, evaluation, deadline] = await Promise.all([
          api.getCollectionSemanticProfile(collection),
          api.getTabularEvaluation().catch(() => null),
          api.getDeadlineReport(collection).catch(() => null),
        ])
        if (cancelled) return
        setSemanticProfile(profile)
        setTabularEval(evaluation)
        setDeadlineReport(deadline)
      } catch (err) {
        if (cancelled) return
        setSemanticError(err instanceof Error ? err.message : 'Falha ao carregar semantica da base.')
        setSemanticProfile(null)
        setDeadlineReport(null)
      } finally {
        if (!cancelled) setSemanticLoading(false)
      }
    }
    void loadSemanticProfile()
    return () => {
      cancelled = true
    }
  }, [activeTab, collection, baseMode, documents])

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

  const workspaceStatusLabel = (() => {
    if (!collection) return 'Nenhuma base selecionada'
    if (hasActiveJobs) return 'Processando documentos'
    if (documents.some((doc) => doc.status === 'indexed')) return 'Base pronta para perguntas'
    if (documents.length > 0) return 'Documentos carregados'
    return 'Aguardando primeiro arquivo'
  })()

  const indexedDocumentsCount = documents.filter((doc) => doc.status === 'indexed').length
  const latestActivityIso = [
    ...documents.map((doc) => doc.updated_at || doc.created_at || ''),
    ...jobs.map((job) => job.finished_at || job.started_at || job.created_at || ''),
  ]
    .filter(Boolean)
    .sort()
    .at(-1)

  const latestActivityLabel = latestActivityIso
    ? `atualizado em ${new Date(latestActivityIso).toLocaleString('pt-BR', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      })}`
    : 'sem atividade recente'

  const baseSummary = {
    statusLabel: workspaceStatusLabel,
    hint: !collection
      ? 'Escolha uma base para iniciar o fluxo de ingestao e conversa.'
      : hasActiveJobs
        ? 'Sua base esta processando arquivos. Ja da para acompanhar o pipeline em tempo real.'
        : indexedDocumentsCount > 0
          ? 'A base esta pronta para responder com apoio dos documentos indexados.'
          : documents.length > 0
            ? 'Os arquivos foram recebidos, mas a indexacao ainda nao terminou.'
            : 'A base existe, mas ainda nao recebeu o primeiro arquivo.',
    docsCount: documents.length,
    indexedCount: indexedDocumentsCount,
    activeJobsCount: jobs.filter((job) =>
      ['queued', 'bronze_received', 'silver_processing', 'silver_extracted', 'gold_indexing', 'processing'].includes(job.status),
    ).length,
    lastUpdatedLabel: latestActivityLabel,
  }

  const timelineSteps = (() => {
    const docsCount = documents.length
    const indexedCount = indexedDocumentsCount
    const activeJobs = baseSummary.activeJobsCount
    const bronzeDone = docsCount > 0
    const silverDone = jobs.some((job) => ['silver_extracted', 'gold_indexing', 'indexed'].includes(job.status)) || indexedCount > 0
    const goldDone = indexedCount > 0

    return [
      {
        id: 'bronze',
        label: 'Bronze',
        meta: bronzeDone ? `${docsCount} arquivo(s) recebidos` : 'recebimento do arquivo bruto e metadados',
        status: bronzeDone ? 'done' : activeJobs > 0 ? 'active' : 'idle',
      },
      {
        id: 'silver',
        label: 'Prata',
        meta: silverDone
          ? 'extracao e normalizacao concluidas'
          : activeJobs > 0
            ? 'extraindo e estruturando conteudo'
            : 'aguardando processamento da camada prata',
        status: silverDone ? 'done' : activeJobs > 0 && bronzeDone ? 'active' : 'idle',
      },
      {
        id: 'gold',
        label: 'Ouro',
        meta: goldDone
          ? `${indexedCount} documento(s) indexados`
          : activeJobs > 0
            ? 'gerando embeddings e indexando'
            : 'aguardando indexacao vetorial',
        status: goldDone ? 'done' : activeJobs > 0 && silverDone ? 'active' : 'idle',
      },
    ] as const
  })()

  const chatStatusBanner = (() => {
    if (!collection) {
      return {
        tone: 'info',
        title: 'Selecione uma base para conversar',
        description: 'Escolha ou crie uma base de documentos antes de iniciar uma conversa.',
        actionLabel: 'Ir para ingestao',
        actionTab: 'ingestion' as const,
        actionTarget: 'pipeline' as const,
      }
    }
    if (hasActiveJobs) {
      return {
        tone: 'info',
        title: 'Base em processamento',
        description: 'Os documentos ainda estao passando pelas camadas Bronze, Prata e Ouro. Algumas respostas podem ficar incompletas ate a indexacao terminar.',
        actionLabel: 'Acompanhar ingestao',
        actionTab: 'ingestion' as const,
        actionTarget: 'jobs' as const,
      }
    }
    if (documents.length === 0) {
      return {
        tone: 'warning',
        title: 'Base sem documentos',
        description: 'Envie o primeiro arquivo para que a base possa responder com contexto.',
        actionLabel: 'Enviar arquivo',
        actionTab: 'ingestion' as const,
        actionTarget: 'pipeline' as const,
      }
    }
    if (indexedDocumentsCount === 0) {
      return {
        tone: 'warning',
        title: 'Indexacao ainda nao concluida',
        description: 'Os arquivos ja foram recebidos, mas o chat ainda depende da conclusao da indexacao.',
        actionLabel: 'Ver pipeline',
        actionTab: 'ingestion' as const,
        actionTarget: 'pipeline' as const,
      }
    }
    return {
      tone: 'success',
      title: 'Base pronta para responder',
      description: `A conversa esta conectada a ${collection} com ${indexedDocumentsCount} documento(s) indexado(s).`,
      actionLabel: '',
      actionTab: 'chat' as const,
      actionTarget: null,
    }
  })()

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
    setSuggestedMode(null)
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
    const targetCollection = collection || 'documentos'
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
          targetCollection,
          embedModel,
          file,
          effectiveIngestProfile,
          tableContext.trim(),
        )
        asyncAccepted = true
        setUploadMsg('ok')
        setUploadError(`Arquivo recebido. Job ${job.id.slice(0, 8)} em andamento.`)
        setTableContextDirty(false)
      } catch (err) {
        const msg = err instanceof Error ? err.message : ''
        const shouldFallback = msg.includes('404') || msg.includes('405')
        if (!shouldFallback) throw err
      }

      if (!asyncAccepted) {
        await api.ingestDocument(targetCollection, embedModel, file, effectiveIngestProfile, tableContext.trim())
        setUploadMsg('ok')
        setUploadError('Documento processado e indexado com sucesso.')
        setTableContextDirty(false)
      }

      if (targetCollection) {
        setCollection(targetCollection)
        setModeStateForCollection(targetCollection, {
          mode: baseMode,
          locked: true,
          locked_at: new Date().toISOString(),
        })
      }

      setFile(null)
      setSuggestedMode(null)
      if (fileRef.current) fileRef.current.value = ''
      const cols = await refreshCollections()
      const match = cols.find((c) => c.collection === targetCollection)
      if (match) {
        setEmbedModel(match.embedding_model)
      }
      const docs = await api.listDocuments(targetCollection).catch(() => [])
      setDocuments(docs)
      if (jobsSupported) {
        const data = await api.listIngestionJobs(40).catch(() => [])
        setJobs(data)
      }
      setSidebarKey((k) => k + 1)
    } catch (err) {
      setUploadMsg('err')
      setUploadError(err instanceof Error ? err.message : 'Erro ao enviar arquivo.')
    } finally {
      setUploading(false)
    }
  }

  const saveTableContext = async () => {
    if (!collection) return
    try {
      const result = await api.updateCollectionContext(collection, tableContext.trim())
      setUploadMsg('ok')
      setUploadError(`Contexto salvo para ${result.updated_documents} documento(s) da base.`)
      setTableContextDirty(false)
      await refreshIngestionData()
    } catch (err) {
      setUploadMsg('err')
      setUploadError(err instanceof Error ? err.message : 'Falha ao salvar contexto da base.')
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
        activeTab={activeTab}
        baseMode={baseMode}
        collection={collection}
        collections={collections}
        documentsCount={documents.length}
        indexedDocumentsCount={indexedDocumentsCount}
        jobsSupported={jobsSupported}
        activeJobsCount={baseSummary.activeJobsCount}
        isModeLocked={isModeLocked}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((s) => !s)}
        onSelectConv={selectConv}
        onSelectTab={setActiveTab}
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
              <div className="topbar-title">Workspace ativo</div>
              <div className="topbar-subtitle">{workspaceStatusLabel}</div>
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
              const selected = e.target.files?.[0] ?? null
              setFile(selected)
              if (selected && !isModeLocked && documents.length === 0) {
                const suggestion = inferSuggestedModeFromFile(selected)
                setSuggestedMode(suggestion)
                if (suggestion) {
                  setBaseMode(suggestion.mode)
                  if (collection) {
                    setModeStateForCollection(collection, { mode: suggestion.mode, locked: false })
                  }
                }
              } else if (!selected) {
                setSuggestedMode(null)
              }
              setUploadMsg(null)
              setUploadError(null)
            }}
          />

          {activeTab === 'chat' ? (
            <>
              <div ref={scrollRef} className="messages-scroll">
                {error && <div className="error-banner">{error}</div>}

                <div className="messages-inner">
                  <div className={`chat-status-banner ${chatStatusBanner.tone}`}>
                    <div className="chat-status-copy">
                      <div className="chat-status-title">{chatStatusBanner.title}</div>
                      <div className="chat-status-description">{chatStatusBanner.description}</div>
                    </div>
                    {chatStatusBanner.actionLabel && (
                      <button
                        type="button"
                        className="chat-status-action"
                        onClick={() => {
                          setActiveTab(chatStatusBanner.actionTab)
                          setIngestionFocusTarget(chatStatusBanner.actionTarget)
                        }}
                      >
                        {chatStatusBanner.actionLabel}
                      </button>
                    )}
                  </div>

                  {!hasMessages ? (
                    <WelcomeScreen
                      collection={collection || 'documentos'}
                      mode={baseMode}
                      onSuggestionClick={(question) => {
                        setInput(question)
                        void send(question)
                      }}
                    />
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
            <IngestionPanel
              baseMode={baseMode}
              isModeLocked={isModeLocked}
              suggestedMode={suggestedMode}
              documents={documents}
              file={file}
              uploading={uploading}
              uploadMsg={uploadMsg}
              uploadError={uploadError}
              tableContext={tableContext}
              tableContextDirty={tableContextDirty}
              jobsSupported={jobsSupported}
              visibleJobs={visibleJobs}
              deletingCollection={deletingCollection}
              deletingDocId={deletingDocId}
              artifactPreview={artifactPreview}
              artifactLoadingId={artifactLoadingId}
              artifactError={artifactError}
              artifactTab={artifactTab}
              semanticProfile={semanticProfile}
              semanticLoading={semanticLoading}
              semanticError={semanticError}
              tabularEval={tabularEval}
              deadlineReport={deadlineReport}
              baseSummary={baseSummary}
              timelineSteps={timelineSteps.map((step) => ({ ...step }))}
              focusTarget={ingestionFocusTarget}
              onFocusHandled={() => setIngestionFocusTarget(null)}
              onRefresh={refreshIngestionData}
              onModeChange={onModeChange}
              onTableContextChange={(value) => {
                setTableContext(value)
                setTableContextDirty(true)
              }}
              onSaveTableContext={saveTableContext}
              onPickFile={() => fileRef.current?.click()}
              onUpload={handleUpload}
              onClearSelectedFile={clearSelectedFile}
              onDeleteCurrentCollection={deleteCurrentCollection}
              onOpenArtifacts={openArtifacts}
              onDeleteDocument={deleteDocument}
              onDownloadArtifact={downloadArtifact}
              onArtifactTabChange={setArtifactTab}
            />
          )}
        </div>
      </main>
    </div>
  )
}
