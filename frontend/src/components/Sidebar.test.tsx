import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import Sidebar from './Sidebar'

vi.mock('../api', () => ({
  api: {
    listConversations: vi.fn().mockResolvedValue([
      {
        id: 'conv-1',
        title: 'Conversa inicial',
        collection: 'rol',
        embedding_model: 'model',
        created_at: '2026-04-07T12:00:00Z',
        updated_at: '2026-04-07T12:00:00Z',
      },
    ]),
    deleteConversation: vi.fn().mockResolvedValue(undefined),
  },
}))

describe('Sidebar', () => {
  it('shows process logic and lets the user switch tabs', async () => {
    const user = userEvent.setup()
    const onSelectTab = vi.fn()

    render(
      <Sidebar
        activeConvId="conv-1"
        activeTab="ingestion"
        baseMode="tabular"
        collection="rol"
        collections={[{ collection: 'rol', embedding_model: 'model' }]}
        documentsCount={2}
        indexedDocumentsCount={1}
        jobsSupported
        activeJobsCount={1}
        isModeLocked
        sidebarOpen
        onToggleSidebar={vi.fn()}
        onSelectConv={vi.fn()}
        onSelectTab={onSelectTab}
        onNewConv={vi.fn()}
        onCollectionChange={vi.fn()}
      />,
    )

    expect(screen.getByText('Estado atual do processo')).toBeInTheDocument()
    expect(screen.getByText('Fluxo da Base')).toBeInTheDocument()
    expect(screen.getByText(/1 job\(s\) em andamento/)).toBeInTheDocument()
    expect(screen.queryByText('Base selecionada')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /expandir fluxo da base/i }))
    expect(screen.getByText('Base selecionada')).toBeInTheDocument()
    expect(screen.getAllByText('Ingestao').length).toBeGreaterThan(0)
    expect(screen.getByText('Processamento')).toBeInTheDocument()
    expect(screen.getByText('Conversa')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /chat/i }))
    expect(onSelectTab).toHaveBeenCalledWith('chat')
    await waitFor(() => {
      expect(screen.getByText('Conversa inicial')).toBeInTheDocument()
    })
  })
})
