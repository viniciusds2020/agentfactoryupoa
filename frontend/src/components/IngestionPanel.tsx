import { useEffect, useRef } from 'react'
import { Download, Loader2, Paperclip, RefreshCw, Trash2, Upload, X } from 'lucide-react'

import type {
  CollectionSemanticProfile,
  DeadlineReport,
  DocumentArtifacts,
  DocumentRecord,
  IngestionJob,
  TabularEvaluation,
} from '../api'
import { MODE_HELP, MODE_LABELS, stageLabel, stageProgress } from '../appModes'
import type { BaseMode, SuggestedMode } from '../appModes'

type ArtifactTab = 'markdown' | 'json'
type TimelineStep = {
  id: string
  label: string
  meta: string
  status: 'done' | 'active' | 'idle'
}

type BaseSummary = {
  statusLabel: string
  hint: string
  docsCount: number
  indexedCount: number
  activeJobsCount: number
  lastUpdatedLabel: string
}

type Props = {
  baseMode: BaseMode
  isModeLocked: boolean
  suggestedMode: SuggestedMode
  documents: DocumentRecord[]
  file: File | null
  uploading: boolean
  uploadMsg: 'ok' | 'err' | null
  uploadError: string | null
  tableContext: string
  tableContextDirty: boolean
  jobsSupported: boolean
  visibleJobs: IngestionJob[]
  deletingCollection: boolean
  deletingDocId: string | null
  artifactPreview: DocumentArtifacts | null
  artifactLoadingId: string | null
  artifactError: string | null
  artifactTab: ArtifactTab
  semanticProfile: CollectionSemanticProfile | null
  semanticLoading: boolean
  semanticError: string | null
  tabularEval: TabularEvaluation | null
  deadlineReport: DeadlineReport | null
  baseSummary: BaseSummary
  timelineSteps: TimelineStep[]
  focusTarget?: 'pipeline' | 'jobs' | 'documents' | null
  onFocusHandled?: () => void
  onRefresh: () => void | Promise<void>
  onModeChange: (mode: BaseMode) => void
  onTableContextChange: (value: string) => void
  onSaveTableContext: () => void | Promise<void>
  onPickFile: () => void
  onUpload: () => void | Promise<void>
  onClearSelectedFile: () => void
  onDeleteCurrentCollection: () => void | Promise<void>
  onOpenArtifacts: (doc: DocumentRecord) => void | Promise<void>
  onDeleteDocument: (doc: DocumentRecord) => void | Promise<void>
  onDownloadArtifact: (format: ArtifactTab) => void | Promise<void>
  onArtifactTabChange: (tab: ArtifactTab) => void
}

export default function IngestionPanel({
  baseMode,
  isModeLocked,
  suggestedMode,
  documents,
  file,
  uploading,
  uploadMsg,
  uploadError,
  tableContext,
  tableContextDirty,
  jobsSupported,
  visibleJobs,
  deletingCollection,
  deletingDocId,
  artifactPreview,
  artifactLoadingId,
  artifactError,
  artifactTab,
  semanticProfile,
  semanticLoading,
  semanticError,
  tabularEval,
  deadlineReport,
  baseSummary,
  timelineSteps,
  focusTarget,
  onFocusHandled,
  onRefresh,
  onModeChange,
  onTableContextChange,
  onSaveTableContext,
  onPickFile,
  onUpload,
  onClearSelectedFile,
  onDeleteCurrentCollection,
  onOpenArtifacts,
  onDeleteDocument,
  onDownloadArtifact,
  onArtifactTabChange,
}: Props) {
  const pipelineRef = useRef<HTMLElement | null>(null)
  const jobsRef = useRef<HTMLElement | null>(null)
  const documentsRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!focusTarget) return
    const refMap = {
      pipeline: pipelineRef,
      jobs: jobsRef,
      documents: documentsRef,
    } as const
    const target = refMap[focusTarget].current
    if (!target) return
    if (typeof target.scrollIntoView === 'function') {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
    target.classList.add('ops-card-focus')
    const timer = window.setTimeout(() => {
      target.classList.remove('ops-card-focus')
      onFocusHandled?.()
    }, 1800)
    return () => window.clearTimeout(timer)
  }, [focusTarget, onFocusHandled])

  return (
    <div className="messages-scroll">
      <div className="messages-inner monitoring-stack">
        <section ref={pipelineRef} className="ops-panel glass-panel ops-card ops-card-pipeline" data-section="pipeline">
          <div className="ops-header">
            <div>
              <div className="ops-kicker">Pipeline</div>
              <div className="ops-title">Upload e Ingestao</div>
            </div>
            <button className="ops-refresh" onClick={() => void onRefresh()}>
              <RefreshCw size={14} />
              atualizar
            </button>
          </div>

          <div className="ops-summary-card">
            <div className="ops-summary-head">
              <div>
                <div className="ops-kicker">Resumo da Base</div>
                <div className="ops-title">{baseSummary.statusLabel}</div>
              </div>
              <span className="top-badge">{baseSummary.lastUpdatedLabel}</span>
            </div>
            <div className="ops-item-meta">{baseSummary.hint}</div>
            <div className="ops-summary-stats">
              <div className="ops-summary-stat">
                <strong>{baseSummary.docsCount}</strong>
                <span>documentos</span>
              </div>
              <div className="ops-summary-stat">
                <strong>{baseSummary.indexedCount}</strong>
                <span>indexados</span>
              </div>
              <div className="ops-summary-stat">
                <strong>{baseSummary.activeJobsCount}</strong>
                <span>jobs ativos</span>
              </div>
            </div>
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
                    {!isModeLocked && suggestedMode && documents.length === 0 && (
                      <div className="ops-item-meta" style={{ marginTop: 8 }}>
                        Sugestao automatica: <strong>{MODE_LABELS[suggestedMode.mode]}</strong>
                        {suggestedMode.subtype === 'catalog' ? ' • subtipo Catalogo/Codigos' : suggestedMode.subtype === 'analytic' ? ' • subtipo Analitico' : ''}
                        {' '}({suggestedMode.reason}).
                      </div>
                    )}
                  </div>
                </div>
                {baseMode === 'tabular' && (
                  <div className="ops-item">
                    <div className="ops-item-main">
                      <div className="ops-item-title">Contexto da Base</div>
                      <div className="ops-item-meta">
                        Descreva o que esta tabela representa e o significado das colunas principais. Isso melhora a interpretacao das perguntas.
                      </div>
                      <textarea
                        value={tableContext}
                        onChange={(e) => onTableContextChange(e.target.value)}
                        placeholder="Ex.: Cadastro de clientes com renda mensal, score de credito, cidade, estado e status do cadastro."
                        rows={4}
                        style={{
                          width: '100%',
                          marginTop: 10,
                          borderRadius: 12,
                          border: '1px solid rgba(76, 201, 240, 0.2)',
                          background: 'rgba(5, 12, 24, 0.72)',
                          color: 'inherit',
                          padding: '12px 14px',
                          resize: 'vertical',
                        }}
                      />
                      {documents.length > 0 && (
                        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10 }}>
                          <button className="empty-upload-btn" onClick={() => void onSaveTableContext()} disabled={uploading || !tableContextDirty}>
                            salvar contexto
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
              {!file ? (
                <button className="empty-upload-btn" onClick={onPickFile}>
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
                      onClick={() => void onUpload()}
                      disabled={uploading}
                    >
                      {uploading ? 'enviando...' : 'iniciar ingestao'}
                    </button>
                    <button className="empty-upload-remove" onClick={onClearSelectedFile}>
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
              <div className="ops-timeline">
                {timelineSteps.map((step, index) => (
                  <div key={step.id} className={`ops-timeline-step ${step.status}`}>
                    <div className="ops-timeline-rail">
                      <div className="ops-timeline-dot">{index + 1}</div>
                      {index < timelineSteps.length - 1 && <div className="ops-timeline-line" />}
                    </div>
                    <div className="ops-timeline-body">
                      <div className="ops-item-title">{step.label}</div>
                      <div className="ops-item-meta">{step.meta}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section ref={jobsRef} className="ops-panel glass-panel ops-card ops-card-jobs" data-section="jobs">
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
              Jobs assinc disponiveis apenas quando a rota enterprise esta habilitada.
            </div>
          )}
        </section>

        {baseMode === 'tabular' && (
          <section className="ops-panel glass-panel ops-card ops-card-docs">
            <div className="ops-header">
              <div>
                <div className="ops-kicker">Semantica</div>
                <div className="ops-title">Perfil da Base</div>
              </div>
              <button className="ops-refresh" onClick={() => void onRefresh()}>
                <RefreshCw size={14} />
                atualizar
              </button>
            </div>
            {semanticLoading ? (
              <div className="ops-empty">Carregando perfil semantico...</div>
            ) : semanticError ? (
              <div className="ops-item-error">{semanticError}</div>
            ) : !semanticProfile || semanticProfile.columns.length === 0 ? (
              <div className="ops-empty">O perfil semantico aparecera apos a ingestao tabular da base.</div>
            ) : (
              <div className="ops-list ops-list-compact">
                <div className="ops-item">
                  <div className="ops-item-main">
                    <div className="ops-item-title">Contexto e Sujeito</div>
                    <div className="ops-item-meta">
                      sujeito: {semanticProfile.profile?.subject_label || 'registros'}
                    </div>
                    <div className="ops-item-meta">
                      tipo detectado: {(
                        semanticProfile.profile?.table_type === 'catalog'
                          ? 'Catalogo/Codigos'
                          : semanticProfile.profile?.table_type === 'time_series'
                            ? 'Serie temporal'
                            : semanticProfile.profile?.table_type === 'transactional'
                              ? 'Transacional'
                              : semanticProfile.profile?.table_type === 'mixed'
                                ? 'Misto'
                                : 'Planilhas/Tabelas analiticas'
                      )}
                    </div>
                    <div className="ops-item-meta">
                      {semanticProfile.profile?.base_context || tableContext || 'Sem contexto salvo.'}
                    </div>
                  </div>
                </div>
                {tabularEval && (
                  <div className="ops-item">
                    <div className="ops-item-main">
                      <div className="ops-item-title">Benchmark Tabular</div>
                      <div className="ops-item-meta">
                        planner: {(tabularEval.summary.tabular_plan_success_rate * 100).toFixed(0)}% •
                        unidade: {(tabularEval.summary.unit_render_accuracy * 100).toFixed(0)}% •
                        schema: {(tabularEval.summary.schema_question_success_rate * 100).toFixed(0)}%
                      </div>
                      <div className="ops-item-meta">
                        dataset: {tabularEval.dataset || 'tabular_gold'} • casos: {tabularEval.cases}
                      </div>
                      {tabularEval.suites && (
                        <div className="ops-item-meta">
                          suites: {Object.keys(tabularEval.suites).join(', ')}
                        </div>
                      )}
                    </div>
                  </div>
                )}
                {deadlineReport && (
                  <div className="ops-item">
                    <div className="ops-item-main">
                      <div className="ops-item-title">Relatorio de Prazos</div>
                      <div className="ops-item-meta">
                        procedimentos analisados: {deadlineReport.total_procedimentos}
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 10 }}>
                        {deadlineReport.faixas.map((faixa) => {
                          const faixaNorm = faixa.faixa.toLowerCase()
                          const bg =
                            faixaNorm.includes('imediato') ? 'rgba(255, 99, 132, 0.16)'
                              : faixaNorm.includes('3 dias') ? 'rgba(255, 159, 64, 0.16)'
                                : faixaNorm.includes('5 dias') || faixaNorm.includes('10 dias') ? 'rgba(255, 205, 86, 0.16)'
                                  : faixaNorm.includes('nao autoriza') ? 'rgba(75, 192, 192, 0.16)'
                                    : 'rgba(201, 203, 207, 0.16)'
                          return (
                            <span key={faixa.faixa} className="top-badge" style={{ background: bg }}>
                              {faixa.faixa}: {faixa.count} ({faixa.pct.toFixed(0)}%)
                            </span>
                          )
                        })}
                      </div>
                      {!!deadlineReport.alertas.length && (
                        <div className="ops-item-meta" style={{ marginTop: 10 }}>
                          alertas: {deadlineReport.alertas.slice(0, 3).map((item) => `${item.codigo} - ${item.titulo}`).join('; ')}
                        </div>
                      )}
                    </div>
                  </div>
                )}
                {semanticProfile.columns.slice(0, 8).map((col) => (
                  <div key={col.column_name} className="ops-item">
                    <div className="ops-item-main">
                      <div className="ops-item-title">
                        {col.column_name} • {col.semantic_type}
                      </div>
                      <div className="ops-item-meta">
                        role: {col.role} • unidade: {col.unit || 'n/a'} • cardinalidade: {col.cardinality}
                      </div>
                      <div className="ops-item-meta">{col.description}</div>
                      <div className="ops-item-meta">
                        aliases: {col.aliases.slice(0, 5).join(', ') || '-'}
                      </div>
                      <div className="ops-item-meta">
                        operacoes: {col.allowed_operations.join(', ') || '-'}
                      </div>
                      {!!semanticProfile.value_catalog?.[col.column_name]?.length && (
                        <div className="ops-item-meta">
                          exemplos reais: {semanticProfile.value_catalog[col.column_name].slice(0, 4).map((item) => item.raw_value).join(', ')}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {semanticProfile.columns.length > 8 && (
                  <div className="ops-empty">
                    Exibindo 8 de {semanticProfile.columns.length} colunas perfiladas.
                  </div>
                )}
              </div>
            )}
          </section>
        )}

        <section ref={documentsRef} className="ops-panel glass-panel ops-card ops-card-docs" data-section="documents">
          <div className="ops-header">
            <div>
              <div className="ops-kicker">Inventario</div>
              <div className="ops-title">Documentos na Base</div>
            </div>
            <button
              className="ops-refresh"
              onClick={() => void onDeleteCurrentCollection()}
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
                      onClick={() => void onOpenArtifacts(doc)}
                      disabled={artifactLoadingId === doc.id}
                      aria-label={`Preview ${doc.filename || doc.doc_id}`}
                      title="Preview Prata (Markdown/JSON)"
                    >
                      {artifactLoadingId === doc.id ? <Loader2 size={14} className="animate-spin" /> : 'P'}
                    </button>
                    <button
                      className="ops-delete"
                      onClick={() => void onDeleteDocument(doc)}
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
                    onClick={() => void onDownloadArtifact('markdown')}
                    disabled={!artifactPreview.markdown_preview}
                    title="Baixar markdown"
                  >
                    <Download size={14} />
                    md
                  </button>
                  <button
                    className="ops-refresh"
                    onClick={() => void onDownloadArtifact('json')}
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
                  onClick={() => onArtifactTabChange('markdown')}
                  disabled={!artifactPreview.markdown_preview}
                >
                  Markdown
                </button>
                <button
                  className={`view-tab ${artifactTab === 'json' ? 'active' : ''}`}
                  onClick={() => onArtifactTabChange('json')}
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
  )
}
