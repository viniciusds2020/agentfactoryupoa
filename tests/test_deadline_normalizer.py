from src.deadline_normalizer import normalize_deadline


def test_normalize_deadline_imediato():
    info = normalize_deadline("Urgencia/Emergencia - imediato")
    assert info.days == 0
    assert info.urgency == "imediato"
    assert info.faixa == "Imediato"


def test_normalize_deadline_10_dias():
    info = normalize_deadline("Atendimento em regime de internacao eletiva - ate 10 (dez) dias uteis")
    assert info.days == 10
    assert info.unit == "dias_uteis"
    assert info.urgency == "medio"
    assert info.faixa == "Ate 10 dias"


def test_normalize_deadline_sem_cobertura():
    info = normalize_deadline("Sem Cobertura")
    assert info.days is None
    assert info.urgency == "sem_cobertura"
    assert info.faixa == "Sem Cobertura"


def test_normalize_deadline_nao_autoriza():
    info = normalize_deadline("Consultas nao autorizam")
    assert info.days is None
    assert info.urgency == "nao_autoriza"
    assert info.faixa == "Nao autoriza"
