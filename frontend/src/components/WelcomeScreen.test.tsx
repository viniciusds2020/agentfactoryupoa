import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import WelcomeScreen from './WelcomeScreen'

describe('WelcomeScreen', () => {
  it('shows tabular suggestions and emits click', async () => {
    const user = userEvent.setup()
    const onSuggestionClick = vi.fn()

    render(
      <WelcomeScreen
        collection="rol"
        mode="tabular"
        onSuggestionClick={onSuggestionClick}
      />,
    )

    const button = screen.getByRole('button', { name: 'Qual a cobertura do procedimento 10049?' })
    await user.click(button)

    expect(screen.getByText('Sugestoes para catalogos e tabelas')).toBeInTheDocument()
    expect(onSuggestionClick).toHaveBeenCalledWith('Qual a cobertura do procedimento 10049?')
  })
})
