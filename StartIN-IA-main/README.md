# StartIn IA — Microserviço de Inteligência Artificial, Auditoria de Matchmaking, RAG e Web Scraping

Microserviço autônomo desenvolvido em **Python** com **FastAPI**, **Uvicorn**, **BeautifulSoup4**, **pypdf** e integração híbrida com modelos LLM via **Groq API** (`llama-3.3-70b-versatile`, com fallback automático para `llama-3.1-8b-instant` e motor determinístico local de alta disponibilidade).

Este documento apresenta a **documentação operacional** e a **Auditoria Técnica Oficial da IA e da Balança de Matchmaking (40% / 30% / 30%)** utilizada em toda a plataforma **StartIn**.

---

## 🔍 1. Auditoria Geral da Arquitetura de IA

Para garantir transparência, reprodutibilidade matemática e ausência de notas infladas ou arbitrárias ("alucinações numéricas" comuns em LLMs puras), a IA da StartIn opera em **três camadas auditáveis**:

1. **Camada 1 — Motor Determinístico Auditado (`compute_audited_match_breakdown` em `IA.py`)**:
   - Calcula matematicamente a pontuação de `0` a `100` pontos a partir dos dados cadastrais reais da Startup, da Oportunidade (Edital, POC, Primeira Compra, Compra ou Evento) ou da Empresa, sem injetar tags fictícias em campos não preenchidos.
   - Aplica rigorosamente a balança **`40% Área de Atuação + 30% Tecnologias + 30% Demais Critérios`**, gerando crescimento progressivo e natural conforme os critérios coincidem.
2. **Camada 2 — Enriquecimento Qualitativo com Trava Matemática via LLM (Groq)**:
   - Quando conectada à API Groq (`temperature=0.15`, saída JSON estruturada), a LLM recebe o dossiê completo e o vetor `audited_scores` pré-calculado pela Camada 1.
   - **Trava de Auditoria (`_recompute_audit_with_llm_semantic`)**: a LLM **não pode sobrescrever livremente o score final**. Ela atua exclusivamente na redação das justificativas, ações recomendadas e no refinamento conceitual da subfaixa semântica (`Problema × Demanda`, limitada estritamente de `0` a `5` pontos). Os pilares de **Área (`0..40`)**, **Tecnologias (`0..30`)**, **Estágio/TRL (`0..10`)**, **Modalidade/Modelo (`0..10`)** e **Localização (`0..5`)** permanecem 100% travados na fórmula auditada.
3. **Camada 3 — Saída em Texto Plano Puro e Rastreabilidade (`auditBreakdown` + 4 Pilares)**:
   - Todos os textos gerados pela IA passam por sanitização determinística (`to_plain_text` e `to_structured_plain_text`), removendo qualquer formatação Markdown (`*`, `**`, `#`, `---`, tabelas `|`, crases ou HTML).
   - Cada item avaliado retorna o objeto `auditBreakdown` completo no JSON da API e expõe a soma exata diretamente nos 4 cards visuais da interface (`Perfil: X/40 pts`, `Requisitos: Y/30 pts`, `Objetivo: Z1/20 pts`, `Contexto: Z2/10 pts`), cuja soma fecha exatamente o percentual exibido (`40 + 30 + 20 + 10 = 100 pts`).

---

## ⚖️ 2. Auditoria da Balança de Matchmaking (40% / 30% / 30%)

### 2.1 Fórmula Matemática Oficial

```text
Score Final (0% a 100%) = Pilar_Area (0 a 40) + Pilar_Tecnologias (0 a 30) + Pilar_Demais (0 a 30)
```

Onde `Pilar_Demais (0 a 30)` é decomposto de forma fixa e auditável em quatro subcritérios objetivos:

| Pilar / Subcritério Auditado | Peso Oficial | Pontuação (`pts`) | Campos Auditados (Startup × Alvo) | Exibição nos 4 Cards da UI (`reasons`) |
|---|---:|---:|---|---|
| **1. Área de Atuação / Setor** | **40%** | `0` a `40` | `mainArea`, `mainSector` × `sector`, `title`, `tags`, `demands`, `description` | `reasons.perfil` (`X/40 pts`) |
| **2. Tecnologias Iguais / Compatíveis** | **30%** | `0` a `30` | `technologies`, `tags`, `areasOfInterest`, `howItWorks` × `tags`, `title`, `description`, `demands` | `reasons.requisitos` (`Y/30 pts`) |
| **3.1 Estágio de Maturidade & TRL NASA** | **10%** | `0` a `10` | `currentStage` (`Validação` / `Tração`), `trlScore` (`1–9`), `tractionAttested` × `type`, `operationMode`, `preferredStage` | `reasons.objetivo` (soma `3.1 + 3.2` -> `Z1/20 pts`) |
| **3.2 Modalidade & Modelo de Negócio** | **10%** | `0` a `10` | `opportunityTypes`, `mainObjective`, `investmentNeed` (`0..6`) + `businessModel` (`0..4`) × modalidade do alvo | `reasons.objetivo` (soma `3.1 + 3.2` -> `Z1/20 pts`) |
| **3.3 Afinidade Semântica Problema × Demanda** | **5%** | `0` a `5` | `problem`, `howItWorks`, `differentiator`, `impact` × `description`, `shortDescription`, `demands` | `reasons.contexto` (soma `3.3 + 3.4` -> `Z2/10 pts`) |
| **3.4 Localização, Região & Formato** | **5%** | `0` a `5` | `location`, `state`, `operationRegion` × `location`, `meta`, `metaInfo`, `deadline` | `reasons.contexto` (soma `3.3 + 3.4` -> `Z2/10 pts`) |
| **TOTAL AUDITADO** | **100%** | **`0` a `100`** | **Crescimento progressivo sem piso mínimo artificial** | **Soma exata dos 4 cards = `matchScore%`** |

---

### 2.2 Pilar 1 — Área de Atuação / Setor (`0` a `40` pontos | Peso 40%)

Antes da comparação, todas as strings passam por normalização Unicode `NFD` sem acentos e em caixa baixa (`_norm_match_text`). O motor classifica Startup e Alvo em **12 clusters setoriais canônicos (`AUDITED_DOMAIN_CLUSTERS`)**:

1. `saude_biotech` (*Saúde & Healthtech*: saúde, healthtech, medtech, biotech, clínica, hospital, telemedicina, prontuário, diagnóstico, fármaco)
2. `financas_seguros` (*Finanças & Fintech*: fintech, finanças, insurtech, banco, pagamentos, crédito, contábil, fiscal, cobrança, open finance)
3. `agro_alimentos` (*Agronegócio & AgTech*: agtech, agro, agronegócio, agrícola, foodtech, pecuária, safra, fazenda, irrigação)
4. `educacao_rh` (*Educação & RH*: edtech, educação, ensino, aprendizagem, hrtech, recursos humanos, treinamento, capacitação)
5. `energia_sustentabilidade` (*Energia, ESG & CleanTech*: cleantech, greentech, energia, solar, eólica, ESG, carbono, resíduos, saneamento, descarbonização)
6. `industria_manufatura` (*Indústria & Manufatura*: indústria, manufatura, automação industrial, fábrica, manutenção preditiva, mineração, óleo e gás)
7. `logistica_mobilidade` (*Logística & Mobilidade*: logtech, logística, transporte, frota, entrega, mobilidade, supply chain, armazém, roteirização)
8. `varejo_consumo` (*Varejo & Consumo*: retailtech, varejo, e-commerce, comércio, marketplace, omnicanal, martech)
9. `governo_cidades` (*Governo & Cidades Inteligentes*: govtech, governo, setor público, smart city, gestão pública, prefeitura, compras públicas)
10. `ciberseguranca_juridico` (*Cibersegurança, Legal & Compliance*: cibersegurança, LGPD, antifraude, legaltech, jurídico, compliance, regtech)
11. `construcao_imobiliario` (*Construção & Imobiliário*: construtech, proptech, construção, imobiliário, obras, infraestrutura)
12. `ia_software_dados` (*IA, Dados & Software Corporativo*: inteligência artificial transversal, ciência de dados, deeptech, software corporativo)

#### Curva Progressiva de Pontuação da Área (`0..40 pts`)
Seja `sub_hits` o número de termos específicos de subdomínio da startup encontrados também no título, tags ou demandas do alvo:

- **Faixa A — Mesma Área Primária (`32` a `40` pts)**: o cluster setorial da startup coincide diretamente com o setor/título/tags primárias do alvo (`primary_domain_overlap`) ou há correspondência direta do rótulo da área (`direct_label_match`).
  - Fórmula: `min(40, 31 + min(9, sub_hits * 2 + bonus_rotulo))`, onde `bonus_rotulo` vale `3` para match direto do nome da área ou `1` para match por cluster. Assim, mesmo dentro da mesma área, a nota cresce naturalmente de `32` até `40` conforme o subdomínio específico (ex: *hospital*, *prontuário*, *clínica*) também coincide.
- **Faixa B — Área Compatível no Escopo/Demanda Detalhada (`23` a `31` pts)**: o alvo pertence a um contexto corporativo amplo, mas menciona explicitamente o domínio da startup na descrição das demandas (`context_domain_overlap`).
  - Fórmula: `min(31, 22 + min(9, sub_hits * 2 + 1))`.
- **Faixa C — Edital / Chamada Pública Multissetorial (`16` a `26` pts)**: editais horizontais de fomento (ex: *FINEP*, *FAPESP PIPE*, *Centelha*, *Sebrae*, *Inovação Aberta Multissetorial*) que aceitam startups de múltiplos setores sem conflito setorial.
  - Fórmula: `min(26, 16 + min(10, sub_hits * 3))` — concede crédito parcial justo por elegibilidade setorial, reservando a faixa `32–40` exclusivamente para oportunidades dedicadas à mesma área de atuação.
- **Faixa D — Áreas Distintas com Vocabulário Adjacente (`6` a `15` pts)**: domínios primários diferentes, mas com `1` ou `2` termos correlatos (`min(15, sub_hits * 6)`).
- **Faixa E — Áreas Distintas sem Interseção (`0` pts)**: nenhuma relação setorial (`0/40`).

---

### 2.3 Pilar 2 — Tecnologias Iguais / Compatíveis (`0` a `30` pontos | Peso 30%)

O motor extrai o conjunto tecnológico real da Startup e do Alvo através de **14 grupos canônicos de sinônimos (`AUDITED_TECH_CANONICAL`)** somados a **tags técnicas customizadas (`custom:<tag>`)**:

- **Grupos Canônicos**: `ia` (*IA, AI, LLM, GenAI, NLP*), `machine_learning` (*ML, Deep Learning, Modelos Preditivos*), `dados_analytics` (*Dados, Data Science, Big Data, BI, Analytics*), `saas` (*SaaS, SaaS B2B, Plataforma Web*), `cloud` (*Cloud, Nuvem, AWS, Azure, GCP, Microsserviços*), `automacao_rpa` (*Automação, RPA, Workflow*), `iot_hardware` (*IoT, Sensores, Embarcados, Edge, Robótica, Telemetria*), `api_integracao` (*APIs, ERP, CRM, Interoperabilidade, HL7, FHIR, Webhook*), `visao_computacional` (*Computer Vision, OCR, Vídeo Analytics*), `ciberseguranca` (*Cybersecurity, Criptografia, Antifraude, Zero Trust, LGPD*), `blockchain_web3` (*Blockchain, Smart Contracts, Tokenização*), `mobile_app` (*Mobile, Android, iOS*), `biotecnologia_lab` (*Bioinformática, Genômica, Nanotecnologia*), `energia_limpa_tech` (*Fotovoltaica, Smart Grid, Hidrogênio Verde, Baterias*).
- **Tags Técnicas Customizadas**: qualquer tecnologia específica fora do dicionário padrão (ex: `5G`, `SAP`, `Gêmeos Digitais`) é normalizada como `custom:<token>` para permitir match exato sem perda de especificidade.
- **Regra Anti-Inflação**: perfis sem tecnologias preenchidas **não** recebem tags padrão artificiais durante o cálculo.

#### Fórmula Progressiva das Tecnologias (`0..30 pts`)
Seja `n_matched` a quantidade de tecnologias coincidentes entre Startup e Alvo, e `req_count` o total de tecnologias exigidas pelo Alvo (`max(1, min(|T_alvo_total|, max(2, |T_alvo_tags|)))`):

1. Se `n_matched == 0` -> **`0` pontos**.
2. Se `n_matched >= 1`:
   - **Parcela de Cobertura dos Requisitos (`0` a `18` pts)**: `coverage_pts = round(18 * min(1.0, n_matched / req_count))`
   - **Parcela Cumulativa de Volume Tecnológico (`5` a `12` pts)**:
     - `1` tecnologia igual -> `+5` pts
     - `2` tecnologias iguais -> `+8` pts
     - `3` tecnologias iguais -> `+10` pts
     - `4+` tecnologias iguais -> `+12` pts
   - **Total do Pilar 2**: `tech_score = min(30, coverage_pts + count_pts)`

Exemplo prático em uma oportunidade que demanda **4 tecnologias** (ex: `IA, Dados, Cloud, APIs`):
- `0` de `4` tecnologias iguais -> **`0 / 30 pts`**
- `1` de `4` tecnologias iguais (`25%` de cobertura) -> `4 + 5` = **`9 / 30 pts`**
- `2` de `4` tecnologias iguais (`50%` de cobertura) -> `9 + 8` = **`17 / 30 pts`**
- `3` de `4` tecnologias iguais (`75%` de cobertura) -> `14 + 10` = **`24 / 30 pts`**
- `4` de `4` tecnologias iguais (`100%` de cobertura) -> `18 + 12` = **`30 / 30 pts`**

---

### 2.4 Pilar 3 — Demais Critérios do Match (`0` a `30` pontos | Peso 30%)

Os **30% restantes** são distribuídos de forma auditada entre quatro dimensões operacionais e estratégicas:

#### 3.1 Estágio de Maturidade & Escala TRL da NASA (`0` a `10` pontos — 10%)
Primeiro, o filtro de elegibilidade de negócio (`filter_opportunities_by_stage`) garante que:
- Startups em **Validação** visualizem `POC`, `Primeira Compra`, `Edital` e `Evento` (ocultando `Compra` de escala pura).
- Startups em **Tração** visualizem `Compra`, `Edital` e `Evento` (ocultando `POC` inicial e `Primeira Compra`).

Em seguida, dentro das oportunidades elegíveis, o subcritério pontua de `0` a `10`:
- **Startup em Tração**:
  - Alvo de `Compra` corporativa ou Empresa buscando `Tração`: base `6 pts` + bônus de maturidade (`+4` se `tractionAttested == True` via PDF ou `TRL >= 8`; `+3` se `TRL == 7`; `+1` se `TRL < 7`) -> **`7` a `10 pts`**.
  - Alvo de `Edital` ou `Evento`: base `6 pts` + (`+3` se tração atestada / `TRL >= 7`, senão `+1`) -> **`7` a `9 pts`**.
  - Alvo exclusivo de validação inicial -> **`3 pts`**.
- **Startup em Validação**:
  - Alvo de `POC`, `Primeira Compra` ou Empresa aberta a `Validação`: base `7 pts` + (`+2` se problema/solução estruturados pela IA) + (`+1` se `TRL >= 3`) -> **`7` a `10 pts`**.
  - Alvo de `Edital` ou `Evento`: base `6 pts` + (`+2` se edital cita P&D/MVP/validação/subvenção, senão `+1`) -> **`7` a `8 pts`**.
  - Alvo exclusivo de escala comercial -> **`2 pts`**.

#### 3.2 Modalidade da Oportunidade & Modelo de Negócio (`0` a `10` pontos — 10%)
- **Aderência de Modalidade e Objetivo (`0` a `6` pts)**:
  - `+4 pts` quando o tipo da oportunidade (`Edital`, `POC`, `Primeira Compra`, `Compra`, `Evento`) consta nas preferências `opportunityTypes` da startup (`+3` quando aberto a todas; `+1` quando divergente).
  - `+2 pts` quando o objetivo estratégico converge com a modalidade (ex: necessidade de investimento/fomento em `Edital`, ou busca de contratos/validação comercial B2B em `POC`/`Compra`/Empresas; `+1` para `Evento`).
- **Aderência de Modelo de Negócio (`0` a `4` pts)**:
  - `+4 pts` para modelos corporativos/escaláveis claramente definidos (`SaaS`, `B2B`, `B2G`, `Enterprise`, `Hardware`, `Licenciamento`); `+2 pts` para outros modelos preenchidos (`B2C`, `Marketplace`); `+1 pt` quando não informado.

#### 3.3 Afinidade Semântica Problema × Demanda / Escopo (`0` a `5` pontos — 5%)
- Extrai radicais semânticos relevantes (`>= 5` caracteres, excluindo `STOPWORDS_MATCH_PT`) dos campos `problem`, `howItWorks`, `differentiator`, `impact` e `mainObjective` da startup e verifica quantos radicais aparecem no escopo/demanda do alvo:
  - `0` radicais -> `0 pts` | `1` radical -> `1 pt` | `2` radicais -> `2 pts` | `3–4` radicais -> `3 pts` | `5–6` radicais -> `4 pts` | `7+` radicais -> `5 pts`.
- Quando a LLM Groq está ativa, ela avalia sinônimos conceituais profundos entre o problema resolvido e a dor do edital/empresa em `semantic_affinity_0_to_5`, sendo combinada por média arredondada com o score determinístico dentro do limite estrito `0..5 pts`.

#### 3.4 Localização, Região de Atuação & Formato (`0` a `5` pontos — 5%)
- Mapeia cidades e as UFs brasileiras (`BRAZIL_STATE_TOKENS`, incluindo agências estaduais como `FAPESP -> SP`, `FACEPE/Porto Digital/Recife -> PE`, `FAPERGS -> RS`, `FAPEMIG -> MG`, etc.):
  - **`5 pts`**: mesma cidade ou mesmo estado/UF entre Startup e Alvo (ex: Startup de `Recife, PE` e oportunidade em `Recife / PE`).
  - **`4 pts`**: oportunidade ou empresa de abrangência `Nacional`, `Brasil`, `Remoto` ou `Online`.
  - **`3 pts`**: abrangência geográfica geral sem restrição conflitante.
  - **`1 pt`**: oportunidade restrita a um estado/UF diferente do estado da startup (ex: Startup de `PE` diante de edital regional exclusivo do `RS` ou `SP`).

---

## 📈 3. Simulação Auditada de Crescimento Natural dos Matches

A tabela abaixo reproduz os valores exatos calculados por `IA.compute_audited_match_breakdown` para uma startup de referência (**Nexa Health** — Área: `Saúde & Healthtech`, Tecnologias: `IA, Dados, Cloud, APIs HL7`, Estágio: `Validação · TRL 5`, Modelo: `SaaS B2B`, Local: `Recife, PE`) diante de **5 oportunidades com níveis crescentes de compatibilidade**:

| Cenário Auditado | Área (`40%`) | Tecnologias (`30%`) | Estágio & TRL (`10%`) | Modalidade & Modelo (`10%`) | Semântica (`5%`) | Localização (`5%`) | Score Final (`100%`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. Edital AgTech Irrigação Rural (RS)** *(outra área, 0 techs iguais, UF restrita distinta)* | `0 / 40` | `0 / 30` *(nenhuma)* | `7 / 10` | `10 / 10` | `0 / 5` | `1 / 5` | **18%** |
| **2. POC Logística Predição de Frota (Brasil)** *(outra área, 1 tech igual: IA)* | `0 / 40` | `9 / 30` *(IA)* | `10 / 10` | `10 / 10` | `1 / 5` | `4 / 5` | **34%** |
| **3. Chamada Pública FINEP Inovação Aberta (Brasil)** *(multissetorial, 2 techs iguais: IA, Cloud)* | `16 / 40` | `17 / 30` *(IA, Cloud)* | `8 / 10` | `10 / 10` | `1 / 5` | `4 / 5` | **56%** |
| **4. POC Rede Hospitalar Nacional (Brasil)** *(mesma área Saúde, 2 techs iguais: IA, Dados)* | `40 / 40` | `17 / 30` *(IA, Dados)* | `10 / 10` | `10 / 10` | `3 / 5` | `4 / 5` | **84%** |
| **5. POC Hospitalar Healthtech Recife (Recife, PE)** *(mesma área, 4/4 techs iguais, mesma cidade/UF)* | `40 / 40` | `30 / 30` *(IA, Dados, Cloud, APIs)* | `10 / 10` | `10 / 10` | `4 / 5` | `5 / 5` | **99%** |

### Exemplo Real do Objeto `auditBreakdown` Retornado pela API

```json
{
  "totalScore": 84,
  "areaScore": 40,
  "areaMax": 40,
  "areaRule": "Mesma área de atuação (Saúde & Healthtech)",
  "techScore": 17,
  "techMax": 30,
  "techRule": "2 tecnologia(s) coincidente(s): Dados & Analytics, IA",
  "matchedTechnologies": ["Dados & Analytics", "IA"],
  "otherScore": 27,
  "otherMax": 30,
  "stageTrlScore": 10,
  "modalityModelScore": 10,
  "problemSemanticScore": 3,
  "locationContextScore": 4,
  "locationRule": "Abrangência nacional / remota compatível",
  "reasons": {
    "perfil": "Área de atuação (40/40 pts): Mesma área de atuação (Saúde & Healthtech) · Modelo SaaS B2B.",
    "requisitos": "Tecnologias (17/30 pts): Coincidências (Dados & Analytics, IA).",
    "objetivo": "Estágio, TRL e Modalidade (20/20 pts): Estágio Validação (TRL 5/9) ideal para POC / Primeira Compra (10/10) + modalidade/modelo (10/10).",
    "contexto": "Problema e Localização (7/10 pts): afinidade de escopo (3/5) + abrangência nacional / remota compatível (4/5)."
  }
}
```

---

## 🧠 4. Auditoria dos Demais Módulos da IA

### 4.1 Matchmaking Triplo Unificado (`40% / 30% / 30%`)
A mesma função `compute_audited_match_breakdown` governa os três fluxos de recomendação da plataforma:
1. **Startup × Oportunidades (`POST /api/ai/match`)**: cruza a startup com editais públicos raspados da web e demandas corporativas salvas no MySQL.
2. **Empresa × Startups (`POST /api/ai/match-startups`)**: avalia todas as startups cadastradas frente ao setor, tags tecnológicas, modalidade (`POC`, `Compra`) e desafios da empresa contratante.
3. **Startup × Empresas (`POST /api/ai/match-companies`)**: ranqueia empresas parceiras aderentes à área de atuação e stack tecnológico da startup.

### 4.2 Web Scraping & RAG com Links Diretos de Editais (`scrape_real_web_opportunities`)
- **Fontes Monitoradas**: `https://inovastartups.com.br/`, `https://fapesp.br/pipe/chamadas` e `http://www.finep.gov.br/chamadas-publicas/chamadaspublicas?situacao=aberta`.
- **Resolução de Deep Links**: o extrator analisa âncoras HTML com `BeautifulSoup4` e resolve URLs completas via `urljoin`, priorizando links diretos para chamadas abertas, páginas específicas de edital e PDFs oficiais (evitando apontar para homepages genéricas).
- **Cache Controlado**: mantém cache em memória com `CACHE_TTL_SECONDS = 3600` (1 hora), podendo ser forçado via `forceRefresh: true`.

### 4.3 Agendador Diário Automático às 08:00 (`_daily_08am_scheduler_loop`)
- Thread daemon iniciada no `lifespan` do FastAPI (`server.py`) que verifica o relógio do sistema a cada 20 segundos.
- Todos os dias às **08:00**, executa `run_daily_scrape_and_match_job(send_to_backend=True)`: raspa novos editais, busca as startups no Spring Boot (`GET /api/startups`), calcula os matches auditados e sincroniza via `POST /api/ai/daily-sync`.

### 4.4 Estruturação de Problema, Auditoria TRL NASA em PDF e Assistente em Texto Plano
- **Estruturação de Solução (`POST /api/ai/enhance-problem`)**: transforma o relato inicial do empreendedor em texto estruturado em texto plano e extrai automaticamente `tags`, `technologies`, `businessModel`, `howItWorks`, `differentiator`, `impact` e `targetAudience` para alimentar a balança `40/30/30`.
- **Auditoria de Tração e Escala TRL NASA (`POST /api/ai/verify-traction-pdf`)**: extrai o conteúdo real do PDF anexado (`pypdf`) do Atestado de Capacidade Técnica, cruza com o questionário TRL (`1` a `9`) da NASA e atesta o estágio de **Tração** quando `TRL >= 7` com evidência documental válida.
- **Assistente Gerador de Templates e Propostas (`POST /api/ai/chat`)**: gera templates completos de submissão, planos de trabalho de 90 dias, textos de mérito tecnológico e e-mails executivos combinando o perfil real da startup com a oportunidade salva em foco e anexos (`PDF`, `DOCX`, `TXT`), estritamente em **texto plano puro (sem Markdown)**.

---

## 📁 5. Estrutura de Arquivos

```text
StartIN-IA-main/
├── IA.py               # Motor auditado 40/30/30, RAG/Scraping, TRL PDF, prompts LLM e filtro texto plano
├── server.py           # Servidor FastAPI, contratos Pydantic (incluindo auditBreakdown) e agendador 08:00
├── requirements.txt    # Dependências Python (fastapi, uvicorn, groq, beautifulsoup4, requests, pypdf)
├── start_ai.bat        # Script de inicialização rápida no Windows (UTF-8, porta 8000)
└── README.md           # Documentação técnica e auditoria oficial da IA e do matchmaking
```

---

## 🚀 6. Instalação, Execução e Verificação da Auditoria

### Instalação e Inicialização
```powershell
pip install -r requirements.txt
python server.py
```
*(Ou execute `.\start_ai.bat` no Windows. O serviço sobe em `http://localhost:8000`.)*

### Comando Rápido para Auditar a Balança no Terminal
Para verificar matematicamente o crescimento progressivo da balança `40% / 30% / 30%` localmente:
```powershell
python -c "import IA; b = IA.compute_audited_match_breakdown({'mainArea':'Saúde & Healthtech','technologies':'IA, Dados','currentStage':'Validação','businessModel':'SaaS B2B','opportunityTypes':'POC','location':'Recife, PE'}, {'title':'POC Hospitalar','sector':'Saúde','type':'POC','tags':'IA, Dados, Cloud, APIs','description':'Triagem clínica com IA','meta':'Brasil'}); print(b)"
```

---

## 📡 7. Endpoints Oficiais da API (`server.py`)

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` , `/health` , `/api/ai/health` | Status do serviço, conexão Groq, pesos oficiais da balança `40/30/30` e status do agendador `08:00` |
| `POST` | `/api/ai/match` | Calcula o match auditado (`0–100%` + `auditBreakdown` + 4 pilares) de uma Startup contra Oportunidades |
| `POST` | `/api/ai/match-startups` | Calcula o match auditado (`40/30/30`) de uma Empresa contra todas as Startups cadastradas |
| `POST` | `/api/ai/match-companies` | Calcula o match auditado (`40/30/30`) de uma Startup contra todas as Empresas cadastradas |
| `POST` | `/api/ai/chat` | Assistente de templates, propostas e dúvidas sobre editais em texto plano puro (suporta anexos PDF/DOCX/TXT) |
| `POST` | `/api/ai/enhance-problem` | Estrutura o problema/solução da startup em texto plano e extrai tecnologias e modelo de negócio |
| `POST` | `/api/ai/verify-traction-pdf` | Audita o PDF de Atestado de Capacidade Técnica junto à escala TRL da NASA (`1–9`) |
| `POST` | `/api/ai/scrape-opportunities` | Executa raspagem sob demanda dos portais oficiais de editais com deep links |
| `GET`/`POST` | `/api/ai/daily-scrape-json` | Dispara manualmente a rotina completa de scraping + matchmaking auditado das 08:00 |
