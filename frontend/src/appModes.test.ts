import { inferSuggestedModeFromFile, loadModeStateMap, saveModeStateMap } from './appModes'

describe('appModes', () => {
  beforeEach(() => {
    let store: Record<string, string> = {}
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        getItem: (key: string) => store[key] ?? null,
        setItem: (key: string, value: string) => {
          store[key] = value
        },
        removeItem: (key: string) => {
          delete store[key]
        },
        clear: () => {
          store = {}
        },
      },
    })
    window.localStorage.removeItem('kb-mode-locks-v1')
  })

  it('suggests tabular catalog mode for procedure pdfs', () => {
    const file = new File(['x'], 'Rol-de-Procedimentos.pdf', { type: 'application/pdf' })
    const suggestion = inferSuggestedModeFromFile(file)

    expect(suggestion).toEqual(
      expect.objectContaining({
        mode: 'tabular',
        subtype: 'catalog',
      }),
    )
  })

  it('persists mode state in localStorage', () => {
    saveModeStateMap({
      base_a: { mode: 'tabular', locked: true, locked_at: '2026-04-06T12:00:00Z' },
    })

    expect(loadModeStateMap()).toEqual({
      base_a: { mode: 'tabular', locked: true, locked_at: '2026-04-06T12:00:00Z' },
    })
  })
})
