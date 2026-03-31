const BASE = '/api/v1'

const STORAGE_KEY = 'agent-factory-workspace-api-key'

let activeWorkspaceApiKey =
  typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_KEY) || '' : ''

function withWorkspaceHeaders(init?: HeadersInit): HeadersInit {
  const headers = new Headers(init)
  if (activeWorkspaceApiKey) headers.set('X-API-Key', activeWorkspaceApiKey)
  return headers
}

export interface Conversation {
  id: string
  title: string
  collection: string
  embedding_model: string
  created_at: string
  updated_at: string
}

export interface Source {
  chunk_id: string
  doc_id: string
  excerpt: string
  score: number
  page_number?: number | null
  source_filename?: string
  citation_label?: string
  source_kind?: string
  query_summary?: string
  result_preview?: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  sources: Source[]
}

export interface ChatResponse {
  conversation_id: string
  answer: string
  sources: Source[]
  request_id: string
}

export interface CollectionInfo {
  collection: string
  embedding_model: string
}

export interface DocumentRecord {
  id: string
  workspace_id: string
  collection: string
  doc_id: string
  filename: string
  embedding_model: string
  status: string
  chunks_indexed: number
  error: string
  context_hint: string
  created_at: string
  updated_at: string
}

export interface IngestionJob {
  id: string
  workspace_id: string
  collection: string
  doc_id: string
  filename: string
  embedding_model: string
  status: string
  chunks_indexed: number
  error: string
  created_at: string
  started_at: string
  finished_at: string
  stage: string
  progress_pct: number
}

export interface DocumentArtifacts {
  doc_id: string
  markdown_path: string
  json_path: string
  markdown_preview: string
  json_preview: string
  available: boolean
}

export interface TableProfile {
  workspace_id: string
  collection: string
  table_name: string
  base_context: string
  subject_label: string
  table_type: string
  created_at: string
  updated_at: string
}

export interface ColumnProfile {
  workspace_id: string
  collection: string
  column_name: string
  display_name: string
  physical_type: string
  semantic_type: string
  role: string
  unit: string
  aliases: string[]
  examples: string[]
  description: string
  cardinality: number
  allowed_operations: string[]
}

export interface ValueCatalogItem {
  normalized_value: string
  raw_value: string
  frequency: number
}

export interface CollectionSemanticProfile {
  collection: string
  profile: TableProfile | null
  columns: ColumnProfile[]
  value_catalog: Record<string, ValueCatalogItem[]>
}

export interface TabularEvaluation {
  cases: number
  summary: Record<string, number>
  details: Array<Record<string, unknown>>
  dataset?: string
  context_hint?: string
  suites?: Record<string, Record<string, unknown>>
}

export interface Workspace {
  id: string
  name: string
  api_key: string
  is_default: boolean
  created_at: string
}

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...options,
    headers: withWorkspaceHeaders(options?.headers),
  })
  if (!res.ok) {
    const contentType = res.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      const payload = (await res.json().catch(() => null)) as { detail?: string } | null
      const detail = payload?.detail?.trim()
      throw new Error(detail || `HTTP ${res.status}`)
    }

    const text = (await res.text()).trim()
    throw new Error(text || `HTTP ${res.status}`)
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

export const api = {
  getWorkspaceApiKey() {
    return activeWorkspaceApiKey
  },

  setWorkspaceApiKey(apiKey: string | null) {
    activeWorkspaceApiKey = apiKey || ''
    if (typeof window !== 'undefined') {
      if (activeWorkspaceApiKey) {
        window.localStorage.setItem(STORAGE_KEY, activeWorkspaceApiKey)
      } else {
        window.localStorage.removeItem(STORAGE_KEY)
      }
    }
  },

  listCollections: () => req<CollectionInfo[]>('/collections/available'),

  listConversations: (q = '') =>
    req<Conversation[]>(`/conversations${q ? `?q=${encodeURIComponent(q)}` : ''}`),

  createConversation: (collection: string, embeddingModel: string) =>
    req<Conversation>('/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection, embedding_model: embeddingModel }),
    }),

  getMessages: (convId: string) => req<Message[]>(`/conversations/${convId}/messages`),

  renameConversation: (convId: string, title: string) =>
    req<Conversation>(`/conversations/${convId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }),

  deleteConversation: (convId: string) => req<void>(`/conversations/${convId}`, { method: 'DELETE' }),

  sendMessage: (
    collection: string,
    embeddingModel: string,
    domainProfile: string,
    question: string,
    history: { role: string; content: string }[],
    conversationId?: string,
  ) =>
    req<ChatResponse>('/chat/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: conversationId ?? null,
        collection,
        embedding_model: embeddingModel,
        domain_profile: domainProfile,
        question,
        history,
      }),
    }),

  sendMessageStream: async (
    collection: string,
    embeddingModel: string,
    domainProfile: string,
    question: string,
    history: { role: string; content: string }[],
    conversationId: string | undefined,
    onSources: (sources: Source[], conversationId: string, requestId: string) => void,
    onToken: (token: string) => void,
    onDone: () => void,
  ) => {
    const res = await fetch(BASE + '/chat/message/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...Object.fromEntries(new Headers(withWorkspaceHeaders()).entries()),
      },
      body: JSON.stringify({
        conversation_id: conversationId ?? null,
        collection,
        embedding_model: embeddingModel,
        domain_profile: domainProfile,
        question,
        history,
      }),
    })
    if (!res.ok) {
      const payload = await res.json().catch(() => null) as { detail?: string } | null
      throw new Error(payload?.detail || `HTTP ${res.status}`)
    }
    const reader = res.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const json = line.slice(6).trim()
        if (!json) continue
        try {
          const event = JSON.parse(json)
          if (event.type === 'sources') {
            onSources(event.sources || [], event.conversation_id || '', event.request_id || '')
          } else if (event.type === 'token') {
            onToken(event.token)
          } else if (event.type === 'done') {
            onDone()
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  },

  ingestDocument: (collection: string, embeddingModel: string, file: File, domainProfile?: string, contextHint?: string) => {
    const form = new FormData()
    form.append('collection', collection)
    form.append('embedding_model', embeddingModel)
    if (domainProfile) form.append('domain_profile', domainProfile)
    if (contextHint) form.append('context_hint', contextHint)
    form.append('file', file)
    return req<{ collection: string; doc_id: string; chunks_indexed: number }>('/ingest', {
      method: 'POST',
      body: form,
    })
  },

  ingestDocumentAsync: (
    collection: string,
    embeddingModel: string,
    file: File,
    domainProfile?: string,
    contextHint?: string,
  ) => {
    const form = new FormData()
    form.append('collection', collection)
    form.append('embedding_model', embeddingModel)
    if (domainProfile) form.append('domain_profile', domainProfile)
    if (contextHint) form.append('context_hint', contextHint)
    form.append('file', file)
    return req<IngestionJob>('/ingest/async', {
      method: 'POST',
      body: form,
    })
  },

  updateCollectionContext: (collection: string, contextHint: string) =>
    req<{ collection: string; context_hint: string; updated_documents: number }>(`/collections/${encodeURIComponent(collection)}/context`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ context_hint: contextHint }),
    }),

  getCollectionSemanticProfile: (collection: string) =>
    req<CollectionSemanticProfile>(`/collections/${encodeURIComponent(collection)}/semantic-profile`),

  listIngestionJobs: (limit = 50) =>
    req<IngestionJob[]>(`/ingest/jobs?limit=${encodeURIComponent(String(limit))}`),

  getIngestionJob: (jobId: string) =>
    req<IngestionJob>(`/ingest/jobs/${encodeURIComponent(jobId)}`),

  listDocuments: (collection?: string) =>
    req<DocumentRecord[]>(`/documents${collection ? `?collection=${encodeURIComponent(collection)}` : ''}`),

  deleteDocument: (docId: string, collection: string, embeddingModel?: string) =>
    req<void>(`/documents/${encodeURIComponent(docId)}?collection=${encodeURIComponent(collection)}${embeddingModel ? `&embedding_model=${encodeURIComponent(embeddingModel)}` : ''}`, {
      method: 'DELETE',
    }),

  getDocumentArtifacts: (docId: string, collection: string, embeddingModel?: string) =>
    req<DocumentArtifacts>(
      `/documents/${encodeURIComponent(docId)}/artifacts?collection=${encodeURIComponent(collection)}${embeddingModel ? `&embedding_model=${encodeURIComponent(embeddingModel)}` : ''}`,
    ),

  downloadArtifact: async (
    docId: string,
    collection: string,
    kind: 'markdown' | 'json',
    embeddingModel?: string,
  ) => {
    const path = `/documents/${encodeURIComponent(docId)}/artifacts/download?collection=${encodeURIComponent(collection)}&kind=${encodeURIComponent(kind)}${embeddingModel ? `&embedding_model=${encodeURIComponent(embeddingModel)}` : ''}`
    const res = await fetch(BASE + path, {
      headers: withWorkspaceHeaders(),
    })
    if (!res.ok) {
      const payload = await res.json().catch(() => null) as { detail?: string } | null
      throw new Error(payload?.detail || `HTTP ${res.status}`)
    }
    return res.blob()
  },

  listWorkspaces: () => req<Workspace[]>('/workspaces'),

  getTabularEvaluation: () => req<TabularEvaluation>('/evaluation/tabular'),

  health: () => req<{ status: string }>('/health'),
}
