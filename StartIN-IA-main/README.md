# StartIn IA — Microserviço de Inteligência Artificial, RAG e Web Scraping

Microserviço autônomo desenvolvido em **Python** com **FastAPI**, **Uvicorn**, **BeautifulSoup4** e integração com modelos LLM via **Groq API** (Llama 3 / fallback semântico de alta disponibilidade). É o cérebro da plataforma **StartIn**, responsável pela raspagem contínua de editais públicos, cálculo de aderência (matchmaking) e pelo assistente virtual inteligente.

---

## 🧠 Principais Capacidades da IA

### 1. Web Scraping & RAG com Links Diretos de Editais
- **Fontes Oficiais Monitoradas**: Portais de inovação e fomento como **FAPESP (PIPE)**, **FINEP**, **Sebrae** e o programa **InovaStartups**.
- **Extração com BeautifulSoup**: A IA varre o HTML das páginas oficiais e extrai título, órgão emissor, escopo temático, datas de vigência, requisitos e os **links diretos para os arquivos oficiais** (ex: `edital-inova-6a-edicao.pdf`), evitando redirecionar o usuário para homepages genéricas.
- **Estruturação JSON**: Converte o texto bruto dos editais em objetos estruturados padronizados contendo títulos claros, descrições resumidas, tags e metadados.

### 2. Agendador Diário Automático às 08:00 (`_daily_08am_scheduler_loop`)
- Uma thread em background monitora continuamente o relógio do servidor.
- Todo dia pontualmente às **08:00 da manhã**, o serviço executa automaticamente:
  1. O scraping dos editais mais recentes na web;
  2. A busca de todas as startups cadastradas no backend Spring Boot (`http://localhost:8080/api/startups`);
  3. O cálculo da compatibilidade (match) de cada startup com os editais disponíveis;
  4. O envio consolidado dos novos editais e notas para o endpoint `/api/ai/daily-sync` do Spring Boot para gravação no MySQL.

### 3. Algoritmo de Matchmaking Multidimensional
O modelo avalia a aderência através de 4 pilares estratégicos:
1. **Perfil da Solução**: Compatibilidade do setor, tecnologias e modelo de negócio (B2B, B2C, SaaS, DeepTech).
2. **Requisitos do Edital**: Estágio de maturidade (Ideação, MVP, Tração, Escala), faturamento e enquadramento geográfico.
3. **Objetivo Estratégico**: Necessidade de recursos (subvenção econômica, contratação, POC, investimento).
4. **Contexto de Mercado**: Tendências do setor e aplicação prática da solução no mercado.

O resultado inclui um **Score de 0 a 100%**, uma **justificativa fundamentada** e uma **recomendação de ação imediata**.

### 4. Matchmaking Bidirecional (Empresas e Startups)
- **Startup x Oportunidades**: Recomenda os editais públicos e chamadas abertas mais compatíveis.
- **Empresa x Startups**: No fluxo corporativo, avalia as startups cadastradas e ranqueia as que melhor atendem aos desafios e POCs lançados pela empresa.
- **Startup x Empresas**: Permite à startup encontrar empresas contratantes com sinergia tecnológica.

### 5. Chatbot Especialista StartIn (`/api/ai/chat`)
- Agente conversacional treinado com diretrizes estritas do ecossistema StartIn.
- Responde de forma sucinta, objetiva e em texto plano, sem desviar do escopo de editais, fomento e conexões corporativas.

---

## 📁 Estrutura de Arquivos

```
StartIN-IA-main/
├── IA.py               # Lógica central: RAG, Web Scraping, prompts LLM e cálculo de pilares
├── server.py           # Servidor FastAPI com rotas REST e agendador em thread daemon
├── requirements.txt    # Dependências Python (fastapi, uvicorn, groq, beautifulsoup4, requests...)
├── start_ai.bat        # Script batch para execução no Windows com codificação UTF-8
└── README.md           # Documentação completa do serviço
```

---

## 🛠 Requisitos e Instalação

### Pré-requisitos
- **Python**: Versão 3.10 ou superior instalada e no PATH do sistema.
- **Conexão com a Internet**: Para a raspagem das fontes oficiais e chamadas à API Groq.

### Passo 1: Instalação das Dependências
Abra o terminal na pasta `StartIN-IA-main` e execute:
```bash
pip install -r requirements.txt
```

### Passo 2: Variáveis de Ambiente (Opcional)
Por padrão, o serviço já possui fallback configurado para a chave da Groq API. Caso deseje utilizar uma chave personalizada, defina a variável de ambiente:
```powershell
$env:GROQ_API_KEY="gsk_sua_chave_groq_aqui"
```

---

## 🚀 Como Executar

### Opção A: Script Rápido (Windows)
Dê um duplo clique ou execute via terminal:
```powershell
.\start_ai.bat
```

### Opção B: Via Python Direto
```powershell
python server.py
```

O microserviço iniciará na porta **`http://localhost:8000`**.

---

## 📡 Endpoints da API FastAPI

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` | Verificação de status e versão do serviço |
| `POST` | `/api/ai/match` | Avalia compatibilidade de uma Startup com uma lista de Oportunidades |
| `POST` | `/api/ai/company-match` | Avalia Startups cadastradas para o perfil de uma Empresa contratante |
| `POST` | `/api/ai/startup-to-companies-match` | Avalia Empresas contratantes para uma Startup |
| `POST` | `/api/ai/chat` | Assistente virtual da StartIn em tempo real |
| `POST` | `/api/ai/daily-job` | Dispara manualmente a rotina de scraping e matchmaking das 08:00 |
| `GET` | `/api/ai/scrape-preview` | Visualiza uma amostra dos editais reais recém-raspados |

---

## 🔗 Integração com o Backend Spring Boot
- **Host do Backend**: `http://localhost:8080`
- **Sincronização Diária**: O script `server.py` envia via `POST /api/ai/daily-sync` um payload com a lista de editais e o mapeamento de notas por startup.
- **Tolerância a Falhas**: Se o backend Spring Boot estiver temporariamente indisponível no momento do cálculo, o microserviço registra o log e mantém os dados em memória para a próxima tentativa.
