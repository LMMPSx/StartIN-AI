# StartIn IA - Microservico de Inteligencia Artificial, Auditoria de Matchmaking, Extracao de PDF, RAG e Web Scraping

Microservico autonomo desenvolvido em Python com FastAPI, Uvicorn, BeautifulSoup4, pypdf e integracao hibrida com modelos LLM via Groq API (`llama-3.3-70b-versatile`, com fallback automatico para `llama-3.1-8b-instant` e motor deterministico local de alta disponibilidade).

Este documento apresenta a documentacao operacional, as novas rotas de extracao de curriculo em PDF, geracao de templates com estilo personalizado, agendamento diferenciado por plano (Free vs Pro) e a Auditoria Tecnica Oficial da Balanca de Matchmaking (40% / 30% / 30%) utilizada em toda a plataforma StartIn.

---

## 1. Arquitetura Geral da IA e Novas Capacidades

Para garantir transparencia, reprodutibilidade matematica e ausencia de notas infladas ou arbitrarias, a IA da StartIn opera em camadas auditaveis e simetricas:

1. **Motor Deterministico Auditado (`compute_audited_match_breakdown` e `_compute_provider_opportunity_compatibility` em `IA.py`)**:
   - Calcula matematicamente a pontuacao de `0` a `100` pontos a partir dos dados reais da Startup, da Empresa, do Prestador de Servicos e da Oportunidade/Demanda.
   - Aplica rigorosamente a balanca **`40% Area de Atuacao + 30% Tecnologias + 30% Contexto e Maturidade`**.
   - **Simetria e Trava de Score (`_recompute_audit_with_llm_semantic`)**: o score final e 100% deterministico e travado na soma matematica dos pilares auditados, garantindo que o percentual exibido nos selos dos cards, na justificativa textual e na tela de detalhes seja sempre identico e simetrico em ambos os lados da conexao (`Empresa <-> Startup` e `Startup em Tracao <-> Prestador / Demanda`).
2. **Extracao Coesa de Curriculo em PDF do Prestador (`extract_provider_resume_pdf` em `IA.py`)**:
   - Le o arquivo PDF anexado pelo prestador de servicos na Etapa 1 do onboarding utilizando `pypdf` (com fallback ASCII/Latin-1 para PDFs compactados).
   - Identifica de forma coesa a categoria principal (`Design`, `Tecnologia`, `Marketing`, `Consultoria`, `Juridico`, `Financeiro`), o servico principal oferecido, as tecnologias e ferramentas dominadas (`toolsAndSkills`), o tempo de experiencia, a localizacao (Cidade, UF), o case/projeto em destaque e redige automaticamente uma **descricao profissional previa em 1a pessoa** pronta para revisao na Etapa 3.
3. **Chatbot Gerador de Templates com Definicao de Estilo (`chat_with_assistant` em `IA.py`)**:
   - Suporta o parametro `templateStyle` enviado pelo usuario no Assistente IA (exclusivo para assinantes do Plano Pro em todos os perfis).
   - Caso o usuario defina um estilo especifico (ex: executivo, tecnico, pitch comercial, formal) ou digite `padrao`, a IA adapta toda a estrutura do documento gerado mantendo coesao com o perfil do usuario e permitindo exportacao direta em **PDF** ou **DOCX** no frontend.
4. **Agendador Automatico Diferenciado por Plano (`_daily_08am_scheduler_loop` em `server.py`)**:
   - Executa todos os dias as **08:00 da manha**:
     - **Plano Pro**: atualizacao e sincronizacao RAG **diaria** as 08:00.
     - **Plano Free**: atualizacao e sincronizacao RAG **semanal** (executada as segundas-feiras as 08:00).

---

## 2. Auditoria da Balanca de Matchmaking (40% / 30% / 30%)

### 2.1. Formula Matematica Oficial e Decomposicao em 10 Subtopicos Tecnicos

```text
Score Final (0% a 100%) = Topico_1_Area (0 a 40) + Topico_2_Tecnologias (0 a 30) + Topico_3_Contexto_Maturidade (0 a 30)
```

O calculo e tecnicamente auditado em 10 subtopicos granulares, retornados individualmente no objeto `subtopics` de cada match:

| Macro-Topico Auditado | Subtopico Tecnico | Pontuacao Max. | Criterio Tecnico e Variaveis Analisadas |
|---|---|---:|---|
| **Topico 1 - Area de Atuacao e Mercado (40 pts)** | **1.1 Dominio Macro-Setorial** | **25 pts** | Sobreposicao entre clusters canonicos (`saude_biotech`, `financas_seguros`, `agro_alimentos`, etc.). Concede `25 pts` para o mesmo cluster primario, `15-18 pts` para escopo corporativo amplo/multissetorial com aderencia contextual e `0 pts` para setores conflitantes. |
| | **1.2 Vocabulario Tecnico & Nicho** | **10 pts** | Frequencia e densidade de termos especializados de nicho encontrados simultaneamente na Startup/Prestador e no Alvo (`sub_hits`), crescendo progressivamente ate `10 pts`. |
| | **1.3 Segmento de Aplicacao & Bonus de Rotulo** | **5 pts** | Match direto e literal do rotulo da area cadastrada (`bonus_rotulo`) ou sinergia de aplicacao especifica no mercado B2B. |
| **Topico 2 - Tecnologias e Stack Tecnica (30 pts)** | **2.1 Cobertura da Stack Essencial** | **18 pts** | Razao matematica de cobertura entre as tecnologias dominadas e as exigidas pelo desafio/edital: `round(18 * min(1.0, n_matched / req_count))`. |
| | **2.2 Densidade Tecnologica / Volume** | **8 pts** | Volume cumulativo de competencias coincidentes: `+3 pts` para 1 tecnologia, `+5 pts` para 2, `+7 pts` para 3, `+8 pts` para 4 ou mais tecnologias em comum. |
| | **2.3 Arquitetura & Integracoes** | **4 pts** | Compatibilidade de modelo arquitetural e camadas de integracao corporativa (APIs REST, Microsservicos, Cloud AWS/GCP/Azure, ERP, Mobile). |
| **Topico 3 - Maturidade, Modalidade e Contexto (30 pts)** | **3.1 Estagio de Maturidade & TRL NASA** | **10 pts** | Validacao vs Tracao: para Validacao, MVP testado (`TRL 4-6`) pontua em POCs e Primeira Compra; para Tracao, sistema homologado (`TRL 7-9`) com Atestado de Capacidade Tecnica pontua em Compras diretas e Demandas de Servico. |
| | **3.2 Modalidade & Modelo de Negocio B2B** | **10 pts** | Convergencia entre a modalidade da oportunidade (`Edital`, `POC`, `Primeira Compra`, `Compra`, `Servico`, `Evento`) e o modelo operacional. |
| | **3.3 Semantica Problema x Demanda** | **5 pts** | Afinidade semantica entre os desafios operacionais descritos e a solucao tecnica ofertada (`0 a 5 pts`). |
| | **3.4 Localizacao, Regiao & Formato** | **5 pts** | Aderencia geografica e de formato: `5 pts` para mesma cidade/UF; `4 pts` para abrangencia Nacional/Remota/Online; `1 pt` para editais com trava regional exclusiva de outro estado. |
| **TOTAL GERAL AUDITADO** | **10 Subtopicos Tecnicos** | **100 pts** | **Soma matematica exata de todos os subtopicos, sem piso ficticio.** |

---

## 3. Estrutura de Arquivos

```text
StartIN-IA-main/
|-- IA.py                   # Motor auditado 40/30/30, extracao de curriculo PDF, TRL PDF, RAG/Scraping e Chatbot
|-- server.py               # Servidor FastAPI, contratos Pydantic e agendador 08:00 (diario Pro / semanal Free)
|-- requirements.txt        # Dependencias Python (fastapi, uvicorn, groq, beautifulsoup4, requests, pypdf)
|-- daily_scrape_08h.json   # Base persistida de editais e eventos raspados
|-- start_ai.bat            # Script de inicializacao rapida no Windows (UTF-8, porta 8000)
`-- README.md               # Documentacao tecnica e auditoria oficial do microservico de IA
```

---

## 4. Instalacao e Execucao

```powershell
cd StartIN-IA-main
pip install -r requirements.txt
python server.py
```
Ou execute `.\start_ai.bat` no Windows. O servico responde em `http://localhost:8000`.

Para validar a sintaxe e compilacao dos modulos Python:
```powershell
python -m py_compile IA.py server.py
```

---

## 5. Endpoints Oficiais da API (`server.py`)

| Metodo | Endpoint | Descricao |
|---|---|---|
| `GET` | `/` , `/health` , `/api/ai/health` | Status do servico, conexao Groq, pesos oficiais da balanca `40/30/30` e status do agendador `08:00` |
| `POST` | `/api/ai/match` | Calcula o match auditado (`0-100%` + `auditBreakdown` + 4 pilares) de uma Startup contra Oportunidades |
| `POST` | `/api/ai/match-startups` | Calcula o match auditado (`40/30/30`) de uma Empresa contra todas as Startups cadastradas |
| `POST` | `/api/ai/match-companies` | Calcula o match auditado (`40/30/30`) de uma Startup contra todas as Empresas cadastradas |
| `POST` | `/api/ai/match-providers` | Calcula o match simetrico entre Startups em Tracao, Demandas Tecnicas e Prestadores de Servicos |
| `POST` | `/api/ai/extract-provider-resume` | Le o curriculo em PDF do prestador e extrai categoria, servico, tecnologias e descricao previa em 1a pessoa |
| `POST` | `/api/ai/enhance-provider-services` | Aprimora em 1a pessoa a descricao profissional do prestador mantendo coesao com o texto editado |
| `POST` | `/api/ai/chat` | Assistente de templates e propostas (Plano Pro) com definicao de estilo (`templateStyle`) e texto plano puro |
| `POST` | `/api/ai/enhance-problem` | Estrutura o problema/solucao da startup ou escopo de demanda tecnica e extrai tags |
| `POST` | `/api/ai/enhance-company-demands` | Estrutura os desafios e demandas operacionais da empresa e extrai tags/topicos |
| `POST` | `/api/ai/verify-traction-pdf` | Audita o PDF de Atestado de Capacidade Tecnica junto a escala TRL da NASA (`1-9`) |
| `POST` | `/api/ai/scrape-opportunities` | Executa raspagem sob demanda dos portais oficiais de editais com deep links |
| `GET`/`POST` | `/api/ai/daily-scrape-json` | Dispara a rotina de scraping e matchmaking auditado das 08:00 (respeitando o filtro de plano Pro/Free) |


---

## Deploy no Vercel (Serverless)

- Root Directory no Vercel: `StartIN-IA-main` (preset `Other`).
- `api/index.py` expoe o `app` FastAPI de `server.py`; o `vercel.json` reescreve todas as rotas para ele com `maxDuration` de 300 s.
- Variaveis de ambiente: `GROQ_API_KEY`, `BACKEND_API_URL` (URL do backend no Railway), opcionais `GROQ_MODEL` e `CRON_SECRET` (modelo em `.env.example`).
- No Vercel o agendador interno fica desligado; a rotina das 08:00 e disparada pelo backend no Railway (`GET /api/ai/daily-scrape-json`) ou por `GET /api/ai/cron/daily`.
- O `daily_scrape_08h.json` e gravado em `/tmp` no Vercel; o arquivo versionado serve de semente inicial.
- Localmente a chave Groq fica em `.env` (ignorado pelo Git).

Passo a passo completo: [../DEPLOY.md](../DEPLOY.md).
