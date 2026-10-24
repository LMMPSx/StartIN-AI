import os
import re
import time
import json
from datetime import datetime
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from groq import Groq
from typing import List, Dict, Any, Optional

# Chave da API Groq e configuração do Backend Java Spring Boot
DEFAULT_GROQ_KEY = "Pão de queijo"
api_key = os.environ.get("GROQ_API_KEY", DEFAULT_GROQ_KEY)
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8080/api")

try:
    client = Groq(api_key=api_key)
except Exception as e:
    print(f"Aviso ao inicializar cliente Groq: {e}")
    client = None

MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Portais reais de Editais, Chamadas Públicas e Eventos de Inovação no Brasil
DEFAULT_PORTAL_URLS = [
    "https://inovastartups.com.br/",
    "https://fapesp.br/pipe/chamadas",
    "http://www.finep.gov.br/chamadas-publicas/chamadaspublicas?situacao=aberta",
]

# Cache em memória das oportunidades raspadas na web
_SCRAPED_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "items": [],
}
CACHE_TTL_SECONDS = 3600  # 1 hora


# =============================================================================
# UTILITÁRIO DE TEXTO PLANO (SEM MARKDOWN) E FILTRO DE ESCOPO DO PROJETO
# =============================================================================

def to_plain_text(text: str) -> str:
    """
    Garante que qualquer resposta do chatbot esteja estritamente em formato
    texto plano (sem markdown) e sucinto.
    """
    if not text:
        return ""

    cleaned = text
    # Remove blocos de código cercados por crases
    cleaned = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("```", ""), cleaned)
    # Converte links markdown [texto](url) em "texto (url)"
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", cleaned)
    # Remove negrito/itálico/tachado/código inline (**, __, *, _, ~~, `)
    cleaned = re.sub(r"(\*\*|__|~~|`)", "", cleaned)
    cleaned = re.sub(r"(?<!\w)\*([^\*\n]+)\*(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", cleaned)
    # Remove cabeçalhos markdown (#, ##, ###)
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
    # Remove marcadores de lista no início de linhas (-, *, +, •) e tabelas (|)
    cleaned = re.sub(r"(?m)^\s*[\-\*\+•]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\|.*\|\s*$", lambda m: " ".join(p.strip() for p in m.group(0).strip("|").split("|") if p.strip()), cleaned)
    cleaned = re.sub(r"\*", "", cleaned)
    cleaned = re.sub(r"#", "", cleaned)
    # Remove excesso de quebras de linha e espaços múltiplos
    cleaned = re.sub(r"\n{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


OFF_TOPIC_KEYWORDS = [
    "receita de", "bolo", "futebol", " escalação", "campeonato", "clima", "previsão do tempo",
    "piada", "horóscopo", "signo", "filme", "série", "novela", "música", "letra de",
    "quem descobriu", "capital da", "capital do", "presidente do", "política", "eleição",
    "jogo de", " traduza ", "poema", "poesia", "história do mundo", "segunda guerra",
    "fórmula de bhaskara", " derivada ", " integral ", "código em c++", "script python para",
]

PROJECT_KEYWORDS = [
    "startin", "startup", "oportunidade", "oportunidades", "edital", "editais",
    "evento", "eventos", "poc", "pocs", "compra", "compras", "match", "matches",
    "aderência", "compatibilidade", "perfil", "fapesp", "finep", "inova", "inovação",
    "fomento", "subvenção", "investimento", "empresa", "cliente", "clientes",
    "dashboard", "configurações", "conta", "onboarding", "preferências", "aplicação",
    "plataforma", "projeto", "sistema", "banco", "buscar", "priorizar", "analisar",
    "olá", "oi", "bom dia", "boa tarde", "boa noite", "ajuda", "como funciona",
]


def is_question_outside_project_scope(user_text: str) -> bool:
    """
    Identifica rapidamente se a mensagem do usuário trata de assuntos gerais
    fora do escopo do projeto StartIn.
    """
    if not user_text or not user_text.strip():
        return False
    lowered = f" {user_text.lower().strip()} "
    if any(kw in lowered for kw in OFF_TOPIC_KEYWORDS) and not any(pk in lowered for pk in PROJECT_KEYWORDS):
        return True
    return False


# =============================================================================
# 1. INTEGRAÇÃO DIRETA COM O BACKEND SPRING BOOT (LEITURA DE USUÁRIO E DADOS)
# =============================================================================

def fetch_user_and_startup_from_backend(
    startup_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Busca diretamente no Backend Java (Spring Boot + MySQL) os dados do usuário
    e todas as informações completas do perfil da startup para realizar o match.
    """
    endpoints = []
    if startup_id:
        endpoints.append(f"{BACKEND_API_URL}/startups/{startup_id}")
    if user_id:
        endpoints.append(f"{BACKEND_API_URL}/startups/user/{user_id}")
    endpoints.append(f"{BACKEND_API_URL}/startups/default")

    for url in endpoints:
        try:
            response = requests.get(url, timeout=8)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"Aviso ao consultar perfil no backend ({url}): {e}")

    return {}


def fetch_all_startups_from_backend() -> List[Dict[str, Any]]:
    """
    Busca todas as startups cadastradas no Backend Java para o processamento diário das 08:00.
    """
    url = f"{BACKEND_API_URL}/startups"
    try:
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data
    except Exception as e:
        print(f"Aviso ao listar todas as startups no backend ({url}): {e}")

    default_profile = fetch_user_and_startup_from_backend()
    return [default_profile] if default_profile else []


def fetch_opportunities_from_backend() -> List[Dict[str, Any]]:
    """
    Busca diretamente no Backend Java (Spring Boot + MySQL) todas as oportunidades
    cadastradas no banco de dados.
    """
    url = f"{BACKEND_API_URL}/opportunities"
    try:
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"Aviso ao consultar oportunidades no backend ({url}): {e}")
    return []


# =============================================================================
# 2. RASPAGEM E EXTRAÇÃO DE EDITAIS E EVENTOS REAIS DA WEB COM IA
# =============================================================================

def fetch_text_from_url(url: str) -> str:
    """
    Faz requisição HTTP na página do portal de editais e realiza RAG estruturado:
    extrai tanto o texto limpo quanto os links específicos (<a href>) de cada edital/chamada/evento
    para que a IA associe a URL exata de cada oportunidade.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "noscript"]):
            element.decompose()

        extracted_links: List[str] = []
        seen_hrefs = set()
        for a_tag in soup.find_all("a", href=True):
            href = (a_tag.get("href") or "").strip()
            link_text = " ".join(a_tag.stripped_strings).strip()
            if not href or href.startswith("#") or href.lower().startswith("javascript:"):
                continue
            full_url = urljoin(url, href)
            if full_url not in seen_hrefs and len(link_text) > 4:
                seen_hrefs.add(full_url)
                extracted_links.append(f"- {link_text}: {full_url}")
            if len(extracted_links) >= 35:
                break

        text = " ".join(soup.stripped_strings)[:4200]
        links_block = "\n".join(extracted_links[:30])
        if links_block:
            return f"{text}\n\n[LINKS ESPECÍFICOS DE EDITAIS/EVENTOS ENCONTRADOS NA PÁGINA ({url})]:\n{links_block}"
        return text
    except Exception as e:
        print(f"Erro ao buscar a URL {url}: {e}")
        return ""


def extract_opportunities_from_text(raw_text: str, url: str) -> List[dict]:
    """
    A IA analisa o texto bruto e o índice RAG de links raspados de um portal real
    (InovaStartups, FAPESP PIPE, FINEP) e extrai até 3 oportunidades/editais/eventos reais
    contidos na página em formato JSON, incluindo a URL específica de cada edital.
    """
    if not raw_text:
        return []

    if not client:
        return [
            {
                "url": url,
                "title": f"Chamada Pública de Inovação ({url.split('//')[-1].split('/')[0]})",
                "type": "EDITAL",
                "focus": raw_text[:300],
                "deadline": "Inscrições abertas · Brasil",
                "tags": "Inovação • Subvenção • Pesquisa",
            }
        ]

    prompt = f"""
    Sua tarefa é realizar RAG sobre o conteúdo e os links reais extraídos do portal de inovação ({url}) e identificar os editais, chamadas públicas, programas de aceleração ou eventos reais mencionados na página.

    CONTEÚDO E LINKS EXTRAÍDOS DO PORTAL ({url}):
    {raw_text}

    Extraia de 1 a 3 oportunidades reais encontradas nesse conteúdo.
    IMPORTANTE SOBRE A URL: Para cada oportunidade, preencha o campo "url" com o LINK ESPECÍFICO da página daquele edital/chamada/evento encontrado na seção [LINKS ESPECÍFICOS DE EDITAIS/EVENTOS ENCONTRADOS NA PÁGINA]. Caso não exista um link específico para o item, utilize "{url}".
    Retorne ESTRITAMENTE um JSON válido com a chave "opportunities" contendo uma lista de objetos no seguinte formato:
    {{
      "opportunities": [
        {{
          "url": "https://link-especifico-do-edital-ou-evento...",
          "title": "Nome real do Edital, Chamada Pública ou Evento encontrado no texto",
          "type": "EDITAL ou EVENTO ou POC",
          "focus": "Descrição detalhada do objetivo do edital/programa, quem pode participar e áreas apoiadas",
          "deadline": "Prazo ou status + Localização (ex: Fluxo Contínuo · Brasil · Remoto)",
          "tags": "3 a 4 tags separadas por ' • ' (ex: Finep • Subvenção • Tecnologia • B2B)"
        }}
      ]
    }}
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um especialista em editais de fomento (FINEP, FAPESP, SEBRAE, InovaStartups). Extraia apenas editais e programas reais do texto fornecido e responda estritamente com JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed = json.loads(completion.choices[0].message.content)
        opps = parsed.get("opportunities")
        if isinstance(opps, list) and len(opps) > 0:
            return opps
    except Exception as e:
        print(f"Erro ao extrair oportunidades com IA para {url}: {e}")

    return [
        {
            "url": url,
            "title": f"Edital de Inovação ({url.split('//')[-1].split('/')[0]})",
            "type": "EDITAL",
            "focus": raw_text[:280],
            "deadline": "Consulte o portal · Brasil",
            "tags": "Edital • Inovação • Fomento",
        }
    ]


def extract_opportunity_from_text(raw_text: str, url: str) -> dict:
    """Mantido por compatibilidade: retorna a primeira oportunidade extraída da URL."""
    items = extract_opportunities_from_text(raw_text, url)
    return items[0] if items else {
        "url": url,
        "title": "Oportunidade de Inovação",
        "type": "EDITAL",
        "focus": raw_text[:250] if raw_text else "Oportunidade capturada da web",
        "deadline": "Fluxo Contínuo",
        "tags": "Inovação • Web",
    }


def scrape_real_web_opportunities(urls: Optional[List[str]] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Raspa os portais oficiais de editais (InovaStartups, FAPESP PIPE, FINEP)
    e retorna a lista de oportunidades reais estruturadas em JSON pela IA.
    """
    now = time.time()
    if not force_refresh and not urls and _SCRAPED_CACHE["items"] and (now - _SCRAPED_CACHE["timestamp"] < CACHE_TTL_SECONDS):
        return _SCRAPED_CACHE["items"]

    target_urls = urls or DEFAULT_PORTAL_URLS
    scraped_list: List[Dict[str, Any]] = []

    for url in target_urls:
        print(f"[StartIn IA] Raspando editais reais em: {url} ...")
        raw_text = fetch_text_from_url(url)
        if not raw_text:
            continue

        extracted_items = extract_opportunities_from_text(raw_text, url)
        for item in extracted_items:
            title = (item.get("title") or "").strip()
            if not title:
                continue
            opp_type = (item.get("type") or "EDITAL").upper()
            if opp_type not in ("EDITAL", "EVENTO", "POC", "COMPRA"):
                opp_type = "EVENTO" if "evento" in opp_type.lower() else "EDITAL"

            scraped_list.append({
                "id": f"web-{len(scraped_list) + 1}",
                "code": f"web-{len(scraped_list) + 1}",
                "title": title,
                "type": opp_type,
                "description": item.get("focus") or item.get("description") or "Edital capturado da web pela IA.",
                "shortDescription": (item.get("focus") or "")[:120],
                "tags": item.get("tags") or "Web • Edital Real • Inovação",
                "url": item.get("url") or url,
                "externalUrl": item.get("url") or url,
                "deadline": item.get("deadline") or "Inscrições abertas · Brasil",
                "meta": item.get("deadline") or "Inscrições abertas · Brasil",
                "metaInfo": item.get("deadline") or "Inscrições abertas · Brasil",
            })

    if not urls and scraped_list:
        _SCRAPED_CACHE["timestamp"] = now
        _SCRAPED_CACHE["items"] = scraped_list

    return scraped_list


# =============================================================================
# 3. MOTOR DE MATCHMAKING ENTRE O USUÁRIO/STARTUP DO BACKEND E OPORTUNIDADES
# =============================================================================

def _format_profile_for_prompt(profile: dict) -> str:
    """Monta um dossiê completo com todos os dados do usuário e as 6 etapas da startup vindos do backend."""
    tags_raw = profile.get("tags") or profile.get("technologies") or []
    if isinstance(tags_raw, list):
        tags_str = ", ".join(str(t) for t in tags_raw if t)
    else:
        tags_str = str(tags_raw or "")

    return f"""
    [DADOS DO USUÁRIO]
    - ID do Usuário: {profile.get('userId') or 'N/A'}
    - Nome do Usuário Responsável: {profile.get('userName') or 'N/A'}
    - E-mail do Usuário: {profile.get('userEmail') or 'N/A'}

    [01. IDENTIDADE DA STARTUP]
    - ID da Startup: {profile.get('id') or 'N/A'}
    - Nome da Startup: {profile.get('name') or 'Não informado'}
    - Área Principal: {profile.get('mainArea') or profile.get('sector') or 'Não informada'}
    - Descrição: {profile.get('description') or 'Não informada'}
    - Localização: {profile.get('location') or 'Não informada'}
    - CNPJ: {profile.get('cnpj') or 'Não informado'}
    - Site: {profile.get('website') or 'Não informado'}
    - Ano de Fundação: {profile.get('foundedYear') or 'Não informado'}

    [02. SOLUÇÃO E TECNOLOGIA]
    - Problema que resolve: {profile.get('problem') or 'Não informado'}
    - Como a solução funciona: {profile.get('howItWorks') or 'Não informado'}
    - Diferencial competitivo: {profile.get('differentiator') or 'Não informado'}
    - Impacto gerado: {profile.get('impact') or 'Não informado'}
    - Tecnologias principais / Tags: {tags_str or 'Não informadas'}
    - Modelo de negócio: {profile.get('businessModel') or 'Não informado'}

    [03. MERCADO]
    - Público-alvo: {profile.get('targetAudience') or 'Não informado'}
    - Setor principal: {profile.get('mainSector') or profile.get('sector') or 'Não informado'}
    - Região de atuação: {profile.get('operationRegion') or 'Não informada'}
    - Estado: {profile.get('state') or 'Não informado'}
    - Perfil de cliente: {profile.get('customerProfile') or 'Não informado'}
    - Tamanho do mercado: {profile.get('marketSize') or 'Não informado'}
    - Concorrentes: {profile.get('competitors') or 'Não informados'}
    - Canal principal: {profile.get('mainChannel') or 'Não informado'}

    [04. MOMENTO E TRAÇÃO]
    - Estágio atual: {profile.get('currentStage') or profile.get('stage') or 'Não informado'}
    - Tração / Clientes: {profile.get('tractionCustomers') or 'Não informado'}
    - Maturidade do produto: {profile.get('productMaturity') or 'Não informada'}
    - Tamanho do time: {profile.get('teamSize') or 'Não informado'}
    - Receita anual: {profile.get('annualRevenue') or 'Não informada'}
    - Crescimento recente: {profile.get('recentGrowth') or 'Não informado'}
    - Investimento captado: {profile.get('investmentRaised') or 'Não informado'}

    [05. OBJETIVOS ESTRATÉGICOS]
    - Principal objetivo: {profile.get('mainObjective') or 'Não informado'}
    - Necessidade de investimento: {profile.get('investmentNeed') or 'Não informada'}
    - Prazo de execução: {profile.get('executionDeadline') or 'Não informado'}
    - Uso dos recursos: {profile.get('resourceUse') or 'Não informado'}
    - Meta para os próximos 12 meses: {profile.get('twelveMonthGoal') or 'Não informada'}
    - Prioridade estratégica: {profile.get('strategicPriority') or 'Não informada'}
    - Parceiros desejados: {profile.get('desiredPartners') or 'Não informados'}

    [06. PREFERÊNCIAS DE MATCHING]
    - Áreas de interesse: {profile.get('areasOfInterest') or 'Não informadas'}
    - Tipos de oportunidade buscados: {profile.get('opportunityTypes') or 'Todos'}
    """


def evaluate_startup_matches(
    startup_profile: Optional[dict] = None,
    opportunities: Optional[List[dict]] = None,
    startup_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> dict:
    """
    Lê os dados reais do usuário e da startup vindos do Backend Spring Boot
    e realiza o cálculo de match contra as oportunidades (do banco + raspadas da web).
    """
    if not startup_profile or (not startup_profile.get("description") and not startup_profile.get("problem")):
        fetched = fetch_user_and_startup_from_backend(
            startup_id=startup_id or (startup_profile.get("id") if startup_profile else None),
            user_id=user_id or (startup_profile.get("userId") if startup_profile else None),
        )
        if fetched:
            if startup_profile:
                merged = {**fetched, **{k: v for k, v in startup_profile.items() if v not in (None, "", [])}}
                startup_profile = merged
            else:
                startup_profile = fetched

    startup_profile = startup_profile or {}

    if not opportunities:
        opportunities = fetch_opportunities_from_backend()

    if not opportunities:
        return {"matches": []}

    opp_lookup = {}
    for o in opportunities:
        oid = str(o.get("id") or o.get("code") or "")
        if oid:
            opp_lookup[oid] = o
        if o.get("title"):
            opp_lookup[o["title"].lower()] = o

    if not client:
        return fallback_evaluate_matches(startup_profile, opportunities)

    profile_dossier = _format_profile_for_prompt(startup_profile)

    prompt = f"""
    Você é o motor oficial de Inteligência Artificial do StartIn.
    Analise TODOS os dados do usuário e de sua startup abaixo (recebidos do banco de dados MySQL via Backend Spring Boot)
    e faça o matchmaking detalhado com CADA UMA das {len(opportunities)} oportunidades listadas (incluindo as oportunidades reais raspadas da web e as do banco).

    {profile_dossier}

    OPORTUNIDADES DISPONÍVEIS (BANCO + WEB):
    {json.dumps(opportunities, ensure_ascii=False, indent=2)}

    INSTRUÇÕES DE AVALIAÇÃO:
    1. Avalie TODAS as {len(opportunities)} oportunidades enviadas e retorne um item em `matches` para cada uma delas.
    2. Calcule um `match_score` (0 a 100) real cruzando:
       - Identidade, área principal e setor da startup
       - Problema resolvido, funcionamento da solução, diferencial, impacto e tecnologias
       - Público-alvo, região/estado, modelo de negócio e maturidade/estágio atual
       - Objetivos de 12 meses, necessidade de investimento e tipos de oportunidade preferidos pelo usuário
    3. Escreva uma `justification` personalizada citando o nome real da startup e por que ela combina com a oportunidade.
    4. Escreva uma `recommended_action` prática para o usuário abordar a oportunidade.
    5. Preencha os 4 pilares obrigatórios em `reasons`:
       - `perfil`: alinhamento entre área/setor/modelo da startup e a oportunidade
       - `requisitos`: aderência das tecnologias, solução e maturidade aos requisitos
       - `objetivo`: convergência entre o objetivo da oportunidade e as metas/necessidade de investimento da startup
       - `contexto`: compatibilidade de localização, estado/região, público-alvo e formato

    Retorne ESTRITAMENTE um JSON válido com esta estrutura:
    {{
      "matches": [
        {{
          "opportunity_id": "id_exato_da_oportunidade",
          "opportunity_title": "Título exato da oportunidade",
          "url": "URL da oportunidade",
          "match_score": 95,
          "justification": "Explicação detalhada baseada nos dados reais do usuário...",
          "recommended_action": "Ação recomendada para esta oportunidade...",
          "reasons": {{
            "perfil": "Resumo do pilar Perfil",
            "requisitos": "Resumo do pilar Requisitos",
            "objetivo": "Resumo do pilar Objetivo",
            "contexto": "Resumo do pilar Contexto"
          }}
        }}
      ]
    }}
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um analista sênior de inovação, venture capital e matchmaking corporativo. Avalie todas as oportunidades fornecidas usando os dados reais do usuário e responda apenas com JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        result = json.loads(completion.choices[0].message.content)
        if result and isinstance(result.get("matches"), list) and len(result["matches"]) > 0:
            for m in result["matches"]:
                oid = str(m.get("opportunity_id") or m.get("opportunityId") or "")
                otitle = (m.get("opportunity_title") or m.get("opportunityTitle") or "").lower()
                src = opp_lookup.get(oid) or opp_lookup.get(otitle)
                if src:
                    m["opportunity_type"] = src.get("type") or "EDITAL"
                    m["opportunity_description"] = src.get("description") or src.get("shortDescription") or ""
                    m["opportunity_meta"] = src.get("meta") or src.get("metaInfo") or src.get("deadline") or "Inscrições abertas · Brasil"
                    m["opportunity_tags"] = src.get("tags") or "Inovação • Tecnologia"
                    if not m.get("url"):
                        m["url"] = src.get("externalUrl") or src.get("url") or ""
            return result
    except Exception as e:
        print(f"Aviso na chamada da Groq para matchmaking: {e}. Executando análise multi-critério com os dados do usuário.")

    return fallback_evaluate_matches(startup_profile, opportunities)


def fallback_evaluate_matches(startup_profile: dict, opportunities: List[dict]) -> dict:
    """
    Motor determinístico multi-critério que analisa TODOS os campos reais do usuário
    e da startup vindos do Backend quando a API externa estiver offline/indisponível.
    """
    name = startup_profile.get("name") or "Sua Startup"
    main_area = startup_profile.get("mainArea") or startup_profile.get("sector") or "Inovação"
    main_sector = startup_profile.get("mainSector") or main_area
    business_model = startup_profile.get("businessModel") or "B2B"
    stage = startup_profile.get("currentStage") or startup_profile.get("stage") or "Validação"
    location = startup_profile.get("location") or startup_profile.get("state") or "Brasil"
    operation_region = startup_profile.get("operationRegion") or "Nacional"
    main_objective = startup_profile.get("mainObjective") or startup_profile.get("twelveMonthGoal") or "Crescimento e validação"
    investment_need = startup_profile.get("investmentNeed") or ""
    problem = startup_profile.get("problem") or startup_profile.get("description") or ""
    how_it_works = startup_profile.get("howItWorks") or ""

    raw_techs = startup_profile.get("technologies") or startup_profile.get("tags") or ""
    if isinstance(raw_techs, list):
        tech_list = [str(t).strip() for t in raw_techs if str(t).strip()]
    else:
        tech_list = [t.strip() for t in str(raw_techs).replace("•", ",").split(",") if t.strip()]

    raw_interests = startup_profile.get("areasOfInterest") or ""
    interest_list = [t.strip() for t in str(raw_interests).replace("•", ",").split(",") if t.strip()]

    raw_opp_types = (startup_profile.get("opportunityTypes") or "").lower()

    all_user_keywords = set(
        w.lower()
        for phrase in (tech_list + interest_list + [main_area, main_sector, business_model, location])
        for w in phrase.replace("/", " ").split()
        if len(w) >= 2
    )

    user_full_text = f"{main_area} {main_sector} {problem} {how_it_works} {' '.join(tech_list)} {' '.join(interest_list)} {main_objective}".lower()

    matches = []
    for opp in opportunities:
        opp_id = str(opp.get("id") or opp.get("code") or "")
        title = opp.get("title") or "Oportunidade"
        opp_type = (opp.get("type") or "EDITAL").upper()
        opp_desc = f"{title} {opp.get('shortDescription') or opp.get('short') or ''} {opp.get('description') or ''} {opp.get('tags') or ''} {opp.get('meta') or opp.get('metaInfo') or ''}".lower()

        score = 74

        if raw_opp_types:
            if opp_type.lower() in raw_opp_types or (opp_type == "EDITAL" and "edita" in raw_opp_types):
                score += 7

        matched_keywords = [kw for kw in all_user_keywords if kw in opp_desc]
        score += min(len(matched_keywords) * 4, 14)

        if any(loc.lower() in opp_desc for loc in [location.split(",")[0].strip(), "brasil", "remoto", operation_region] if loc):
            score += 4

        if opp_type == "EDITAL" and (investment_need or "investimento" in user_full_text or "recursos" in user_full_text):
            score += 4
        elif opp_type in ("POC", "COMPRA") and ("b2b" in user_full_text or "clientes" in user_full_text or "escala" in user_full_text):
            score += 4

        score = min(max(score, 70), 98)

        tech_display = " · ".join(tech_list[:4]) if tech_list else main_area
        interest_display = " · ".join(interest_list[:4]) if interest_list else tech_display

        matches.append({
            "opportunity_id": opp_id,
            "opportunity_title": title,
            "opportunity_type": opp_type,
            "opportunity_description": opp.get("description") or opp.get("shortDescription") or "",
            "opportunity_meta": opp.get("meta") or opp.get("metaInfo") or opp.get("deadline") or "Inscrições abertas · Brasil",
            "opportunity_tags": opp.get("tags") or "Inovação • Tecnologia",
            "url": opp.get("externalUrl") or opp.get("url") or "",
            "match_score": score,
            "justification": (
                f"A startup {name} ({main_area} · {business_model}, em estágio de {stage}) "
                f"possui forte aderência com '{title}' pelas competências em {tech_display} e foco em {main_objective.lower()}."
            ),
            "recommended_action": (
                f"Destacar a solução da {name} ({how_it_works[:90] + '...' if len(how_it_works) > 90 else (how_it_works or 'tecnologia proprietária')}) "
                f"e os resultados no setor de {main_sector}."
            ),
            "reasons": {
                "perfil": f"{main_area} · {business_model} · Setor {main_sector} ({stage})",
                "requisitos": f"Tecnologias da {name}: {interest_display}",
                "objetivo": f"{main_objective}" + (f" · Captação: {investment_need}" if investment_need else ""),
                "contexto": f"{location} · Atuação {operation_region} · {opp.get('meta') or opp.get('metaInfo') or 'Compatível'}",
            },
        })

    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return {"matches": matches}


# =============================================================================
# 4. ROTINA AUTOMÁTICA DIÁRIA ÀS 08:00 DA MANHÃ (SCRAPING + CÁLCULO DE MATCH -> JSON PARA O BANCO)
# =============================================================================

def run_daily_scrape_and_match_job(send_to_backend: bool = True) -> Dict[str, Any]:
    """
    Executa a rotina diária das 08:00 da manhã:
    1. Faz o web scraping de editais/oportunidades nos portais oficiais.
    2. Lê todas as startups e oportunidades do banco MySQL via Backend Spring Boot.
    3. Calcula os matches de cada startup.
    4. Monta o JSON estruturado e o envia para o Backend salvar no MySQL (`POST /api/ai/daily-sync`).
    """
    print(f"[StartIn IA - 08:00 Job] Iniciando raspagem diária e cálculo de matches em {datetime.now().isoformat()}...")
    scraped_opportunities = scrape_real_web_opportunities(force_refresh=True)
    db_opportunities = fetch_opportunities_from_backend()

    combined_opps = list(db_opportunities)
    existing_titles = {str(o.get("title", "")).strip().lower() for o in combined_opps}
    for web_opp in scraped_opportunities:
        if web_opp["title"].strip().lower() not in existing_titles:
            combined_opps.append(web_opp)
            existing_titles.add(web_opp["title"].strip().lower())

    startups = fetch_all_startups_from_backend()
    startup_matches_payload = []

    for st in startups:
        sid = st.get("id")
        sname = st.get("name") or "Startup"
        eval_res = evaluate_startup_matches(startup_profile=st, opportunities=combined_opps)
        raw_matches = eval_res.get("matches", [])

        formatted_matches = []
        for m in raw_matches:
            raw_reasons = m.get("reasons") or {}
            formatted_matches.append({
                "opportunityId": str(m.get("opportunity_id") or m.get("opportunityId") or ""),
                "opportunityTitle": m.get("opportunity_title") or m.get("opportunityTitle") or "Oportunidade",
                "opportunityType": m.get("opportunity_type") or m.get("opportunityType") or "EDITAL",
                "opportunityDescription": m.get("opportunity_description") or m.get("opportunityDescription") or "",
                "opportunityMeta": m.get("opportunity_meta") or m.get("opportunityMeta") or "Inscrições abertas · Brasil",
                "opportunityTags": m.get("opportunity_tags") or m.get("opportunityTags") or "Inovação • Tecnologia",
                "url": m.get("url") or "",
                "matchScore": int(m.get("match_score") or m.get("matchScore") or 80),
                "justification": m.get("justification") or "Perfil compatível com os critérios da oportunidade.",
                "recommendedAction": m.get("recommended_action") or m.get("recommendedAction") or "Submeter candidatura.",
                "reasons": {
                    "perfil": raw_reasons.get("perfil", "Alinhamento com segmento e mercado"),
                    "requisitos": raw_reasons.get("requisitos", "Requisitos técnicos atendidos"),
                    "objetivo": raw_reasons.get("objetivo", "Objetivo estratégico convergente"),
                    "contexto": raw_reasons.get("contexto", "Modelo operacional compatível"),
                },
            })

        startup_matches_payload.append({
            "startupId": sid,
            "startupName": sname,
            "totalAnalyzed": len(combined_opps),
            "matches": formatted_matches,
        })

    daily_json = {
        "executedAt": datetime.now().isoformat(),
        "scheduledTime": "08:00",
        "scrapedOpportunities": scraped_opportunities,
        "startupMatches": startup_matches_payload,
    }

    # Salva snapshot local do JSON gerado às 08h
    try:
        snapshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_scrape_08h.json")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(daily_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Aviso ao salvar snapshot JSON local: {e}")

    # Envia o JSON diretamente para o Backend Spring Boot persistir no banco MySQL
    if send_to_backend:
        sync_url = f"{BACKEND_API_URL}/ai/daily-sync"
        try:
            resp = requests.post(sync_url, json=daily_json, timeout=25)
            print(f"[StartIn IA - 08:00 Job] JSON enviado para o banco via Backend ({sync_url}) -> Status {resp.status_code}")
        except Exception as e:
            print(f"Aviso ao sincronizar JSON das 08:00 com o Backend ({sync_url}): {e}")

    return daily_json


# =============================================================================
# 5. ASSISTENTE IA (RESTRITO AO PROJETO STARTIN, SUCINTO E TEXTO PLANO SEM MARKDOWN)
# =============================================================================

def generate_assistant_chat(
    messages: list,
    startup_profile: Optional[dict] = None,
    opportunities: Optional[List[dict]] = None,
) -> str:
    """
    Gera resposta para o Assistente IA do StartIn:
    - Fala SOMENTE sobre o projeto/aplicação StartIn (perfil da startup, oportunidades, editais, eventos, POCs, compras e matches).
    - Recusa perguntas gerais que não tenham relação com a aplicação.
    - Retorna texto sucinto e estritamente em formato TEXTO PLANO (sem markdown).
    """
    last_user_msg = ""
    for msg in reversed(messages or []):
        if msg.get("from") not in ("ai", "assistant"):
            last_user_msg = (msg.get("text") or msg.get("content") or "").strip()
            if last_user_msg:
                break

    if is_question_outside_project_scope(last_user_msg):
        return (
            "Sou o assistente exclusivo da plataforma StartIn e respondo apenas perguntas sobre o projeto, "
            "o perfil da sua startup, editais, eventos, POCs, compras e seus matches de oportunidades."
        )

    if not startup_profile or not startup_profile.get("name"):
        startup_profile = fetch_user_and_startup_from_backend(
            startup_id=startup_profile.get("id") if startup_profile else None,
            user_id=startup_profile.get("userId") if startup_profile else None,
        )

    if not opportunities:
        opportunities = fetch_opportunities_from_backend()

    web_opportunities = scrape_real_web_opportunities()

    all_opps = list(opportunities or [])
    existing_titles = {str(o.get("title", "")).lower() for o in all_opps}
    for w in web_opportunities:
        if str(w.get("title", "")).lower() not in existing_titles:
            all_opps.append(w)

    profile_dossier = _format_profile_for_prompt(startup_profile or {})
    opps_summary = "\n".join(
        f"Tipo: {o.get('type')} | Titulo: {o.get('title')} | Prazo: {o.get('meta') or o.get('metaInfo') or o.get('deadline') or 'Aberto'} | Resumo: {o.get('description') or o.get('shortDescription') or ''}"
        for o in all_opps[:12]
    )

    if not client:
        startup_name = (startup_profile or {}).get("name") or "sua startup"
        opps_titles = ", ".join(o.get("title", "") for o in all_opps[:3])
        return to_plain_text(
            f"Com base no perfil da {startup_name} na plataforma StartIn, existem {len(all_opps)} oportunidades mapeadas, "
            f"incluindo {opps_titles}. Consulte a aba Para voce ou Matches para visualizar a aderencia detalhada."
        )

    system_instruction = f"""
    Voce e o Assistente IA exclusivo da aplicacao StartIn.

    REGRAS OBRIGATORIAS E INVIOLAVEIS:
    1. ESCOPO ESTRITO DO PROJETO: Voce deve falar SOMENTE sobre o projeto e aplicacao StartIn, o perfil da startup do usuario, as oportunidades da plataforma (editais, eventos, POCs e compras diretas) e os matches calculados.
    2. RECUSA DE PERGUNTAS GERAIS: NUNCA responda perguntas gerais, conhecimentos gerais, matematica, receitas, esportes, politica, entretenimento, programacao generica ou qualquer assunto fora da aplicacao StartIn. Se a pergunta do usuario nao tiver relacao direta com a aplicacao StartIn e suas oportunidades/matches, responda apenas: Sou o assistente exclusivo da plataforma StartIn e respondo somente sobre o projeto, o perfil da sua startup, oportunidades e matches.
    3. TEXTO SUCINTO: Sua resposta deve ser curta, direta e sucinta, tendo no maximo 3 a 4 frases em um unico paragrafo.
    4. FORMATO TEXTO PLANO (SEM MARKDOWN): Responda ESTRITAMENTE em texto plano puro. E proibido usar asteriscos, negrito, italico, hashtags, titulos, listas com tracos ou marcadores, tabelas ou blocos de codigo.

    DADOS REAIS DA STARTUP DO USUARIO NO PROJETO STARTIN:
    {profile_dossier}

    OPORTUNIDADES E MATCHES ATUAIS NO STARTIN:
    {opps_summary}
    """

    chat_history = [{"role": "system", "content": system_instruction}]
    for msg in (messages or [])[-6:]:
        role = "assistant" if msg.get("from") in ["ai", "assistant"] else "user"
        content = msg.get("text") or msg.get("content") or ""
        chat_history.append({"role": role, "content": content})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_history,
            temperature=0.2,
            max_tokens=260,
        )
        raw_reply = completion.choices[0].message.content or ""
        return to_plain_text(raw_reply)
    except Exception as e:
        print(f"Erro no chat da IA: {e}")
        startup_name = (startup_profile or {}).get("name") or "sua startup"
        opps_titles = ", ".join(o.get("title", "") for o in all_opps[:3])
        return to_plain_text(
            f"Analisei o perfil da {startup_name} na plataforma StartIn e identifiquei oportunidades alinhadas como {opps_titles}. "
            f"Acesse as abas Para voce e Matches para acompanhar os detalhes e prazos."
        )


if __name__ == "__main__":
    print("=== StartIn IA: Executando rotina de raspagem e cálculo de matches (JSON) ===")
    resultado_diario = run_daily_scrape_and_match_job(send_to_backend=True)
    print(json.dumps(resultado_diario, indent=2, ensure_ascii=False))
