export type BaseMode = 'general' | 'legal' | 'tabular'

export type SuggestedMode = {
  mode: BaseMode
  subtype?: 'catalog' | 'analytic'
  reason: string
} | null

export type BaseModeState = {
  mode: BaseMode
  locked: boolean
  locked_at?: string
}

const MODE_STORAGE_KEY = 'kb-mode-locks-v1'

export const MODE_LABELS: Record<BaseMode, string> = {
  general: 'Conversa Geral',
  legal: 'Juridico/Contratos',
  tabular: 'Planilhas/Tabelas',
}

export const MODE_HELP: Record<BaseMode, string> = {
  general: 'Bom para documentos corporativos, politicas e comunicados.',
  legal: 'Prioriza estrutura juridica para contratos, estatutos e regras formais.',
  tabular: 'Foco em tabelas, planilhas e dados estruturados.',
}

export const MODE_TO_INGEST_PROFILE: Record<BaseMode, string> = {
  general: 'general',
  legal: 'legal',
  tabular: 'tabular',
}

export const MODE_TO_CHAT_PROFILE: Record<BaseMode, string> = {
  general: 'general',
  legal: 'legal',
  tabular: 'tabular',
}

export function loadModeStateMap(): Record<string, BaseModeState> {
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

export function saveModeStateMap(map: Record<string, BaseModeState>) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(MODE_STORAGE_KEY, JSON.stringify(map))
}

export function stageLabel(status: string) {
  if (status === 'queued') return 'Fila (Bronze)'
  if (status === 'bronze_received') return 'Bronze recebido'
  if (status === 'silver_processing') return 'Prata processando'
  if (status === 'processing') return 'Processando'
  if (status === 'indexed') return 'Ouro indexado'
  if (status === 'failed') return 'Falhou'
  return status
}

export function stageProgress(status: string) {
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

export function inferSuggestedModeFromFile(file: File): SuggestedMode {
  const name = file.name.toLowerCase()
  const ext = name.includes('.') ? name.slice(name.lastIndexOf('.')) : ''
  if (['.csv', '.xlsx', '.xls'].includes(ext)) {
    const isCatalog = /(rol|procedimento|procedimentos|codigo|codigos|cobertura|autorizacao|tabela)/.test(name)
    return {
      mode: 'tabular',
      subtype: isCatalog ? 'catalog' : 'analytic',
      reason: isCatalog ? 'arquivo tabular com cara de catalogo/codigos' : 'arquivo tabular estruturado',
    }
  }
  if (/(estatuto|contrato|regulamento|aditivo|juridic|clausula)/.test(name)) {
    return { mode: 'legal', reason: 'nome do arquivo sugere conteudo juridico/contratual' }
  }
  if (/(procedimento|procedimentos|rol|codigo|cobertura|autorizacao|tabela)/.test(name)) {
    return { mode: 'tabular', subtype: 'catalog', reason: 'arquivo sugere catalogo de codigos ou tabela operacional' }
  }
  return { mode: 'general', reason: 'documento narrativo/corporativo por padrao' }
}
