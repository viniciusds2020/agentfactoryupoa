"""Create realistic test PDFs for end-to-end pipeline validation."""
from fpdf import FPDF
import os

os.makedirs("data/docs", exist_ok=True)


def _add_header(pdf, label):
    pdf.add_page()
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, f"{label} | Pagina {pdf.page_no()}", align="R")
    pdf.ln(10)


def _title(pdf, text):
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, text, ln=True, align="C")
    pdf.ln(5)


def _section(pdf, text):
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, text, ln=True)
    pdf.ln(3)


def _body(pdf, text):
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, text)
    pdf.ln(2)


# ── PDF 1: Estatuto Social (juridico, ~8 paginas) ──────────────────────

def create_estatuto():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    h = "Estatuto Social - Cooperativa Exemplo Ltda."

    _add_header(pdf, h)
    _title(pdf, "ESTATUTO SOCIAL")
    _title(pdf, "COOPERATIVA EXEMPLO LTDA.")
    pdf.ln(10)
    _section(pdf, "TITULO I - DA DENOMINACAO, SEDE E DURACAO")
    _section(pdf, "CAPITULO I - DISPOSICOES GERAIS")
    _body(pdf, "Art. 1. A Cooperativa Exemplo Ltda., doravante denominada Cooperativa, e uma sociedade cooperativa de credito, constituida em conformidade com a Lei n. 5.764/71 e demais disposicoes legais aplicaveis.")
    _body(pdf, "Art. 2. A Cooperativa tem sede e foro na cidade de Porto Alegre, Estado do Rio Grande do Sul, na Rua das Cooperativas, n. 500, Bairro Centro, CEP 90000-000.")
    _body(pdf, "Art. 3. O prazo de duracao da Cooperativa e indeterminado.")
    _body(pdf, "Art. 4. A area de atuacao da Cooperativa abrange todo o territorio do Estado do Rio Grande do Sul, podendo expandir-se para outros estados mediante aprovacao em Assembleia Geral.")

    _add_header(pdf, h)
    _section(pdf, "CAPITULO II - DO OBJETO SOCIAL")
    _body(pdf, "Art. 5. A Cooperativa tem por objeto social a prestacao de servicos financeiros aos seus cooperados, incluindo:")
    _body(pdf, "I - concessao de credito pessoal e empresarial;\nII - captacao de depositos a vista e a prazo;\nIII - realizacao de operacoes de credito rural;\nIV - prestacao de servicos de cobranca e recebimentos;\nV - comercializacao de seguros e planos de previdencia.")
    _body(pdf, "Art. 6. A Cooperativa podera celebrar convenios com instituicoes financeiras para ampliar os servicos oferecidos aos cooperados, desde que aprovados pelo Conselho de Administracao.")

    _add_header(pdf, h)
    _section(pdf, "TITULO II - DOS COOPERADOS")
    _section(pdf, "CAPITULO III - DA ADMISSAO")
    _body(pdf, "Art. 7. Podem associar-se a Cooperativa as pessoas fisicas maiores de 18 anos e as pessoas juridicas que tenham compatibilidade com os objetivos sociais e preencham os seguintes requisitos:")
    _body(pdf, "I - residir ou ter sede na area de atuacao da Cooperativa;\nII - apresentar proposta de admissao ao Conselho de Administracao;\nIII - subscrever e integralizar o numero minimo de quotas-partes fixado neste Estatuto;\nIV - nao ter sido eliminado ou excluido de outra cooperativa nos ultimos 5 anos.")
    _body(pdf, "Art. 8. Nao podem associar-se a Cooperativa:")
    _body(pdf, "I - pessoas que exercam cargo politico-partidario eletivo em qualquer esfera;\nII - pessoas com impedimento legal decorrente de condenacao criminal transitada em julgado;\nIII - pessoas que mantenham relacao de emprego com a Cooperativa.")
    _body(pdf, "Paragrafo Unico. A vedacao do inciso I cessa com o termino do mandato ou a renuncia ao cargo.")

    _add_header(pdf, h)
    _section(pdf, "CAPITULO IV - DOS DIREITOS E DEVERES")
    _body(pdf, "Art. 9. Sao direitos do cooperado:\nI - votar e ser votado nas assembleias gerais;\nII - participar das atividades sociais e economicas da Cooperativa;\nIII - receber retorno proporcional as operacoes realizadas no exercicio;\nIV - solicitar informacoes sobre a gestao e as financas da Cooperativa.")
    _body(pdf, "Art. 10. Sao deveres do cooperado:\nI - cumprir as disposicoes deste Estatuto, dos regulamentos e das decisoes assembleares;\nII - integralizar as quotas-partes subscritas nos prazos estabelecidos;\nIII - zelar pelo patrimonio moral e material da Cooperativa;\nIV - comunicar a Cooperativa qualquer alteracao em seus dados cadastrais no prazo de 30 dias.")

    _add_header(pdf, h)
    _section(pdf, "TITULO III - DO CAPITAL SOCIAL")
    _section(pdf, "CAPITULO V - DAS QUOTAS-PARTES")
    _body(pdf, "Art. 11. O capital social da Cooperativa e formado por quotas-partes de valor unitario de R$ 100,00 (cem reais), sendo o minimo de 10 (dez) quotas-partes por cooperado.")
    _body(pdf, "Paragrafo 1. O valor das quotas-partes sera atualizado anualmente com base no INPC/IBGE.")
    _body(pdf, "Paragrafo 2. A integralizacao das quotas-partes podera ser feita a vista ou em ate 12 parcelas mensais.")
    _body(pdf, "Art. 12. O capital social minimo da Cooperativa e de R$ 500.000,00 (quinhentos mil reais), representado por 5.000 quotas-partes.")
    _body(pdf, "Art. 13. A restituicao de quotas-partes ao cooperado que se desligar da Cooperativa sera realizada no prazo maximo de 180 dias, mediante autorizacao do Conselho de Administracao.")

    _add_header(pdf, h)
    _section(pdf, "TITULO IV - DA ADMINISTRACAO")
    _section(pdf, "CAPITULO VI - DA ASSEMBLEIA GERAL")
    _body(pdf, "Art. 14. A Assembleia Geral dos cooperados e o orgao supremo da Cooperativa, com poderes para decidir sobre todos os negocios sociais e tomar as resolucoes que julgar convenientes.")
    _body(pdf, "Art. 15. A Assembleia Geral sera convocada pelo Presidente do Conselho de Administracao, com antecedencia minima de 10 dias, mediante edital publicado em jornal de grande circulacao e afixado nas dependencias da Cooperativa.")
    _body(pdf, "Art. 16. A Assembleia Geral Ordinaria sera realizada uma vez por ano, nos primeiros tres meses apos o encerramento do exercicio social, para deliberar sobre:\nI - prestacao de contas do Conselho de Administracao;\nII - destinacao das sobras ou rateio das perdas;\nIII - eleicao dos membros do Conselho de Administracao e do Conselho Fiscal;\nIV - fixacao da remuneracao dos conselheiros.")

    _add_header(pdf, h)
    _section(pdf, "CAPITULO VII - DO CONSELHO DE ADMINISTRACAO")
    _body(pdf, "Art. 17. O Conselho de Administracao sera composto por 7 (sete) membros efetivos e 3 (tres) suplentes, eleitos pela Assembleia Geral para mandato de 4 (quatro) anos, permitida a reeleicao.")
    _body(pdf, "Art. 18. Compete ao Conselho de Administracao:\nI - aprovar politicas de credito e investimentos;\nII - nomear e destituir o Diretor Executivo;\nIII - autorizar a abertura de filiais e postos de atendimento;\nIV - deliberar sobre a admissao, demissao e exclusao de cooperados;\nV - aprovar o plano anual de atividades e orcamento.")
    _body(pdf, "Art. 19. O Presidente do Conselho de Administracao e o representante legal da Cooperativa, cabendo-lhe convocar e presidir as assembleias gerais e as reunioes do Conselho.")

    _add_header(pdf, h)
    _section(pdf, "CAPITULO VIII - DO CONSELHO FISCAL")
    _body(pdf, "Art. 20. O Conselho Fiscal sera composto por 3 (tres) membros efetivos e 3 (tres) suplentes, eleitos pela Assembleia Geral para mandato de 3 (tres) anos.")
    _body(pdf, "Art. 21. Compete ao Conselho Fiscal:\nI - fiscalizar a administracao da Cooperativa;\nII - examinar balancetes mensais e demonstracoes financeiras;\nIII - emitir parecer sobre as contas anuais do Conselho de Administracao;\nIV - denunciar irregularidades a Assembleia Geral.")
    _body(pdf, "Art. 22. Este Estatuto entra em vigor na data de seu registro no orgao competente, revogando-se todas as disposicoes em contrario.")
    _body(pdf, "Porto Alegre, 15 de janeiro de 2024.")

    pdf.output("data/docs/estatuto_cooperativa.pdf")
    print(f"estatuto_cooperativa.pdf criado - {pdf.page_no()} paginas")


# ── PDF 2: Politica de RH (~4 paginas) ──────────────────────────────────

def create_politica_rh():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    h = "Politica de Gestao de Pessoas - Empresa XYZ S.A."

    _add_header(pdf, h)
    _title(pdf, "POLITICA DE GESTAO DE PESSOAS")
    _title(pdf, "EMPRESA XYZ S.A.")
    pdf.ln(5)
    _body(pdf, "1. OBJETIVO\nEsta politica estabelece as diretrizes e procedimentos para a gestao de pessoas na Empresa XYZ S.A., abrangendo admissao, beneficios, jornada de trabalho, ferias, licencas e desligamento.")
    _body(pdf, "2. ABRANGENCIA\nAplica-se a todos os colaboradores contratados sob regime CLT, estagiarios e aprendizes.")
    _body(pdf, "3. ADMISSAO\n3.1. A admissao de novos colaboradores sera precedida de processo seletivo conduzido pela area de Recursos Humanos, em conjunto com o gestor da area requisitante.")
    _body(pdf, "3.2. Documentos obrigatorios para admissao: CTPS, RG, CPF, comprovante de residencia, certidao de nascimento ou casamento, titulo de eleitor, certificado de reservista (sexo masculino), carteira de vacinacao dos dependentes menores de 7 anos.")
    _body(pdf, "3.3. O contrato de experiencia tera duracao de 90 dias, podendo ser prorrogado uma unica vez, desde que o total nao exceda 90 dias.")

    _add_header(pdf, h)
    _body(pdf, "4. JORNADA DE TRABALHO\n4.1. A jornada padrao e de 8 horas diarias e 44 horas semanais, com intervalo minimo de 1 hora para refeicao.")
    _body(pdf, "4.2. Horas extras deverao ser previamente autorizadas pelo gestor imediato e serao remuneradas com adicional de 50% sobre a hora normal em dias uteis e 100% em domingos e feriados.")
    _body(pdf, "4.3. O banco de horas podera ser utilizado como alternativa ao pagamento de horas extras, devendo ser compensado no prazo maximo de 6 meses.")
    _body(pdf, "4.4. O controle de ponto e obrigatorio para todos os colaboradores, realizado por sistema eletronico biometrico.")
    _body(pdf, "5. BENEFICIOS\n5.1. A empresa oferece os seguintes beneficios:\na) Vale-transporte: conforme legislacao vigente, com desconto de ate 6% do salario base;\nb) Vale-refeicao: R$ 35,00 por dia util trabalhado;\nc) Plano de saude: cobertura nacional, extensivo a dependentes, com coparticipacao de 20%;\nd) Plano odontologico: cobertura basica, sem coparticipacao;\ne) Seguro de vida: cobertura de 24 salarios, sem custo para o colaborador;\nf) Participacao nos Lucros e Resultados (PLR): conforme acordo coletivo vigente.")

    _add_header(pdf, h)
    _body(pdf, "6. FERIAS\n6.1. O colaborador tera direito a 30 dias de ferias apos cada periodo aquisitivo de 12 meses.")
    _body(pdf, "6.2. As ferias poderao ser fracionadas em ate 3 periodos, sendo que um deles nao podera ser inferior a 14 dias corridos e os demais nao inferiores a 5 dias corridos cada.")
    _body(pdf, "6.3. O pagamento das ferias sera realizado ate 2 dias antes do inicio do periodo de gozo.")
    _body(pdf, "6.4. O colaborador podera converter 1/3 das ferias em abono pecuniario, mediante solicitacao com antecedencia minima de 15 dias antes do termino do periodo aquisitivo.")
    _body(pdf, "7. LICENCAS E AFASTAMENTOS\n7.1. Licenca-maternidade: 180 dias (programa Empresa Cidada).\n7.2. Licenca-paternidade: 20 dias (programa Empresa Cidada).\n7.3. Licenca por falecimento de familiar: 3 dias consecutivos para conjuge, pais e filhos; 1 dia para demais familiares.\n7.4. Licenca para casamento: 3 dias consecutivos a partir da data do evento.\n7.5. Afastamento por doenca: a empresa complementa o beneficio previdenciario ate o 30 dia de afastamento, garantindo a remuneracao integral.")

    _add_header(pdf, h)
    _body(pdf, "8. DESLIGAMENTO\n8.1. O desligamento podera ocorrer por iniciativa do colaborador ou da empresa, observando-se as disposicoes da CLT e deste regulamento.")
    _body(pdf, "8.2. O aviso previo sera de 30 dias, acrescido de 3 dias para cada ano completo de servico, ate o maximo de 90 dias.")
    _body(pdf, "8.3. Na rescisao sem justa causa, o colaborador tera direito a: saldo de salario, ferias proporcionais + 1/3, 13 salario proporcional, multa de 40% do FGTS, liberacao do FGTS e guias para seguro-desemprego.")
    _body(pdf, "8.4. O prazo para pagamento das verbas rescisorias e de 10 dias uteis a contar do termino do contrato.")
    _body(pdf, "9. DISPOSICOES FINAIS\n9.1. Esta politica sera revisada anualmente pela Diretoria de Recursos Humanos.\n9.2. Casos omissos serao resolvidos pela Diretoria de RH em conjunto com a Diretoria Juridica.\n9.3. Esta politica entra em vigor na data de sua publicacao, revogando-se todas as disposicoes em contrario.")
    _body(pdf, "Sao Paulo, 01 de marco de 2024.\nDiretoria de Recursos Humanos - Empresa XYZ S.A.")

    pdf.output("data/docs/politica_rh.pdf")
    print(f"politica_rh.pdf criado - {pdf.page_no()} paginas")


if __name__ == "__main__":
    create_estatuto()
    create_politica_rh()
