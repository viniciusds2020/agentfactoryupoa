import type { ComponentProps } from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { CollectionSemanticProfile, DeadlineReport, DocumentArtifacts, DocumentRecord, IngestionJob, TabularEvaluation } from '../api'
import IngestionPanel from './IngestionPanel'

function makeDoc(overrides: Partial<DocumentRecord> = {}): DocumentRecord {
  return {
    id: 'doc-1',
    workspace_id: 'default',
    collection: 'rol',
    doc_id: 'rol',
    filename: 'Rol.pdf',
    embedding_model: 'model',
    status: 'indexed',
    chunks_indexed: 42,
    error: '',
    context_hint: '',
    created_at: '2026-04-06T12:00:00Z',
    updated_at: '2026-04-06T12:00:00Z',
    ...overrides,
  }
}

function makeProps(overrides: Partial<ComponentProps<typeof IngestionPanel>> = {}): ComponentProps<typeof IngestionPanel> {
  const semanticProfile: CollectionSemanticProfile = {
    collection: 'rol',
    profile: {
      workspace_id: 'default',
      collection: 'rol',
      table_name: 'rol_records',
      base_context: 'Catalogo de procedimentos',
      subject_label: 'procedimentos',
      table_type: 'catalog',
      created_at: '2026-04-06T12:00:00Z',
      updated_at: '2026-04-06T12:00:00Z',
    },
    columns: [
      {
        workspace_id: 'default',
        collection: 'rol',
        column_name: 'procedimento',
        display_name: 'Procedimento',
        physical_type: 'text',
        semantic_type: 'identifier',
        role: 'identifier',
        unit: '',
        aliases: ['codigo'],
        examples: [],
        description: 'Codigo do procedimento',
        cardinality: 10,
        allowed_operations: ['lookup'],
      },
    ],
    value_catalog: {
      procedimento: [{ normalized_value: '10049', raw_value: '10049', frequency: 1 }],
    },
  }

  const deadlineReport: DeadlineReport = {
    collection: 'rol',
    total_procedimentos: 3,
    faixas: [{ faixa: 'Imediato', count: 1, pct: 33.3 }],
    alertas: [{ codigo: '10101039', titulo: 'Consulta em pronto socorro', prazo: 'Urgencia/Emergencia - imediato' }],
  }

  const artifactPreview: DocumentArtifacts = {
    doc_id: 'rol',
    markdown_path: 'a.md',
    json_path: 'a.json',
    markdown_preview: '# preview',
    json_preview: '{"ok": true}',
    available: true,
  }

  const tabularEval: TabularEvaluation = {
    cases: 4,
    summary: {
      tabular_plan_success_rate: 1,
      unit_render_accuracy: 1,
      schema_question_success_rate: 1,
    },
    details: [],
    dataset: 'tabular_gold',
  }

  const jobs: IngestionJob[] = [{
    id: 'job-1',
    workspace_id: 'default',
    collection: 'rol',
    doc_id: 'rol',
    filename: 'Rol.pdf',
    embedding_model: 'model',
    status: 'indexed',
    chunks_indexed: 42,
    error: '',
    created_at: '2026-04-06T12:00:00Z',
    started_at: '2026-04-06T12:00:00Z',
    finished_at: '2026-04-06T12:05:00Z',
    stage: 'gold_indexing',
    progress_pct: 100,
  }]

  return {
    baseMode: 'tabular',
    isModeLocked: false,
    suggestedMode: { mode: 'tabular', subtype: 'catalog', reason: 'arquivo tabular com cara de catalogo/codigos' },
    documents: [makeDoc()],
    file: new File(['pdf'], 'Rol.pdf', { type: 'application/pdf' }),
    uploading: false,
    uploadMsg: null,
    uploadError: null,
    tableContext: 'Base de procedimentos',
    tableContextDirty: true,
    jobsSupported: true,
    visibleJobs: jobs,
    deletingCollection: false,
    deletingDocId: null,
    artifactPreview,
    artifactLoadingId: null,
    artifactError: null,
    artifactTab: 'markdown',
    semanticProfile,
    semanticLoading: false,
    semanticError: null,
    tabularEval,
    deadlineReport,
    baseSummary: {
      statusLabel: 'Base pronta para perguntas',
      hint: 'A base esta pronta para responder com apoio dos documentos indexados.',
      docsCount: 1,
      indexedCount: 1,
      activeJobsCount: 0,
      lastUpdatedLabel: 'atualizado em 06/04 12:00',
    },
    timelineSteps: [
      { id: 'bronze', label: 'Bronze', meta: '1 arquivo(s) recebidos', status: 'done' },
      { id: 'silver', label: 'Prata', meta: 'extracao e normalizacao concluidas', status: 'done' },
      { id: 'gold', label: 'Ouro', meta: '1 documento(s) indexados', status: 'done' },
    ],
    focusTarget: null,
    onFocusHandled: vi.fn(),
    onRefresh: vi.fn(),
    onModeChange: vi.fn(),
    onTableContextChange: vi.fn(),
    onSaveTableContext: vi.fn(),
    onPickFile: vi.fn(),
    onUpload: vi.fn(),
    onClearSelectedFile: vi.fn(),
    onDeleteCurrentCollection: vi.fn(),
    onOpenArtifacts: vi.fn(),
    onDeleteDocument: vi.fn(),
    onDownloadArtifact: vi.fn(),
    onArtifactTabChange: vi.fn(),
    ...overrides,
  }
}

describe('IngestionPanel', () => {
  it('renders semantic and deadline sections for tabular bases', () => {
    render(<IngestionPanel {...makeProps()} />)

    expect(screen.getByText('Perfil da Base')).toBeInTheDocument()
    expect(screen.getByText('Resumo da Base')).toBeInTheDocument()
    expect(screen.getByText('Base pronta para perguntas')).toBeInTheDocument()
    expect(screen.getByText('Bronze')).toBeInTheDocument()
    expect(screen.getAllByText('Prata').length).toBeGreaterThan(0)
    expect(screen.getByText('Ouro')).toBeInTheDocument()
    expect(screen.getByText(/Catalogo\/Codigos/)).toBeInTheDocument()
    expect(screen.getByText('Relatorio de Prazos')).toBeInTheDocument()
    expect(screen.getByText(/procedimentos analisados: 3/)).toBeInTheDocument()
    expect(screen.getByText(/exemplos reais: 10049/)).toBeInTheDocument()
  })

  it('calls actions from the upload and artifact controls', async () => {
    const user = userEvent.setup()
    const onSaveTableContext = vi.fn()
    const onUpload = vi.fn()
    const onDownloadArtifact = vi.fn()

    render(
      <IngestionPanel
        {...makeProps({
          onSaveTableContext,
          onUpload,
          onDownloadArtifact,
        })}
      />,
    )

    await user.click(screen.getByRole('button', { name: /salvar contexto/i }))
    await user.click(screen.getByRole('button', { name: /iniciar ingestao/i }))
    await user.click(screen.getByRole('button', { name: /md/i }))

    expect(onSaveTableContext).toHaveBeenCalled()
    expect(onUpload).toHaveBeenCalled()
    expect(onDownloadArtifact).toHaveBeenCalledWith('markdown')
  })

  it('focuses the requested section when asked by the chat banner flow', () => {
    render(
      <IngestionPanel
        {...makeProps({
          focusTarget: 'jobs',
        })}
      />,
    )

    const jobsSection = document.querySelector('[data-section="jobs"]')
    expect(jobsSection).toHaveClass('ops-card-focus')
  })
})
