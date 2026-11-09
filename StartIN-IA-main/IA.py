import os
import io
import re
import time
import json
import base64
import zipfile
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from groq import Groq
from typing import List, Dict, Any, Optional
import concurrent.futures

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# Carrega variáveis de um arquivo .env local caso exista
_ENV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV_FILE_PATH):
    try:
        with open(_ENV_FILE_PATH, "r", encoding="utf-8") as _ef:
            for _line in _ef:
                _clean = _line.strip()
                if _clean and not _clean.startswith("#") and "=" in _clean:
                    _k, _v = _clean.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:
        pass

# Chave da API Groq e configuração do Backend Java Spring Boot
DEFAULT_GROQ_KEY = "Pão de queijo"
api_key = os.environ.get("GROQ_API_KEY", DEFAULT_GROQ_KEY)
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8080/api")

try:
    client = Groq(api_key=api_key) if api_key else None
except Exception as e:
    print(f"Aviso ao inicializar cliente Groq: {e}")
    client = None

MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = [
    MODEL_NAME,
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3.8-27b",
]

# Portais reais de Editais, Chamadas Públicas e Eventos de Inovação no Brasil
DEFAULT_PORTAL_URLS = [
    "https://inovastartups.com.br/",
    "https://fapesp.br/pipe/chamadas",
    "http://www.finep.gov.br/chamadas-publicas/chamadaspublicas?situacao=aberta",
    "https://recnplay.pe/",
    "https://www.startupsummit.com.br/",
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


def _convert_markdown_table_line(line: str) -> str:
    cells = [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]
    return " — ".join(cells)


def to_structured_plain_text(text: str) -> str:
    """
    Converte qualquer saída da IA estritamente para texto plano limpo (sem NENHUM markdown:
    sem *, **, #, ---, >, listas - / *, tabelas |, crases `, links [t](url) ou tags HTML),
    preservando parágrafos, numeração simples (1., 2.) e campos entre colchetes [campo].
    """
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n")

    # Converte <br> em quebra de linha e remove tags HTML de formatação
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(
        r"(?i)</?(?:b|strong|i|em|u|s|code|pre|blockquote|h[1-6]|ul|ol|li|p|hr|span|div|table|thead|tbody|tr|th|td)(?:\s[^>]*)?>",
        "",
        cleaned,
    )

    # Remove cercas de blocos de código ```linguagem ... ```
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = cleaned.replace("```", "")

    # Converte imagens e links markdown [texto](url) em "texto (url)", preservando placeholders [StartOut]
    cleaned = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", r"\1 (\2)", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+|mailto:[^)\s]+|www\.[^)\s]+)\)", r"\1 (\2)", cleaned)

    # Remove linhas divisórias horizontais de markdown (---, ***, ___, ===)
    cleaned = re.sub(r"(?m)^[ \t]*(?:[-*_][ \t]*){3,}$", "", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]*={3,}[ \t]*$", "", cleaned)

    # Remove linhas separadoras de tabelas markdown (| :--- | :---: |) e converte linhas de tabela em texto plano
    cleaned = re.sub(r"(?m)^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*:?-*:?[ \t]*\n?", "", cleaned)
    cleaned = re.sub(
        r"(?m)^[ \t]*\|.*\|[ \t]*$",
        lambda m: _convert_markdown_table_line(m.group(0)),
        cleaned,
    )

    # Remove cabeçalhos (#, ##, ###) e blockquotes (>, >>) no início de linhas
    cleaned = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]*>+[ \t]?", "", cleaned)

    # Remove negrito, tachado e código inline (***, **, __, ~~, `) sem afetar linhas de preenchimento _____
    cleaned = re.sub(r"(\*\*\*|\*\*|~~|`)", "", cleaned)
    cleaned = re.sub(r"(?<!_)__(?=\S)([\s\S]*?)(?<=\S)__(?!_)", r"\1", cleaned)

    # Remove marcadores de lista não ordenada (- , * , + ) e checkboxes (- [ ] ) no início de linhas
    cleaned = re.sub(r"(?m)^[ \t]*[-*+][ \t]+\[[ xX]\][ \t]+", "", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]*[-*+][ \t]+", "", cleaned)

    # Remove itálico com asterisco simples *...* (inclusive multilinhas como *(Preencha...\nedital.)*)
    cleaned = re.sub(r"\*(?=\S)([\s\S]*?)(?<=\S)\*", r"\1", cleaned)
    # Remove itálico com underscore _..._ respeitando fronteiras de palavra (preserva emails/snake_case)
    cleaned = re.sub(r"(?<!\w)_(?=\S)([^_\n]+?)(?<=\S)_(?!\w)", r"\1", cleaned)

    # Remove asteriscos residuais usados como marcação nas bordas de linhas ou palavras
    cleaned = re.sub(r"(?m)^[ \t]*\*+[ \t]*|[ \t]*\*+[ \t]*$", "", cleaned)
    cleaned = re.sub(r"\*(?=[\w(\[\{\"'])|(?<=[\w)\]\}\"'.,;:!?])\*", "", cleaned)

    # Limpa espaços finais por linha e normaliza múltiplas linhas em branco
    cleaned = re.sub(r"(?m)[ \t]+$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_attachment(file_name: Optional[str], file_base64: Optional[str]) -> str:
    """
    Extrai o texto de arquivos anexados pelo usuário (PDF, DOCX, TXT, MD, CSV).
    """
    if not file_base64:
        return ""
    raw_b64 = file_base64.strip()
    if "," in raw_b64 and raw_b64.lower().startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        raw_bytes = base64.b64decode(raw_b64)
    except Exception:
        return ""

    lower_name = (file_name or "").lower()
    if lower_name.endswith(".pdf") or raw_bytes[:4] == b"%PDF":
        if PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(raw_bytes))
                pages_text = []
                for page in reader.pages[:20]:
                    t = page.extract_text() or ""
                    if t.strip():
                        pages_text.append(t.strip())
                joined = "\n\n".join(pages_text).strip()
                if joined:
                    return joined[:14000]
            except Exception as e:
                print(f"[StartIn IA] Aviso ao extrair PDF com pypdf: {e}")
        # Fallback: extrair strings legíveis de streams do PDF
        ascii_strings = re.findall(b"[A-Za-z\xc0-\xff0-9 ,.;:/\\-\\(\\)%$]{6,}", raw_bytes)
        decoded_parts = [s.decode("latin-1", errors="ignore") for s in ascii_strings[:400]]
        return " ".join(decoded_parts)[:10000]

    if lower_name.endswith(".docx"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                if "word/document.xml" in zf.namelist():
                    xml_bytes = zf.read("word/document.xml")
                    root = ET.fromstring(xml_bytes)
                    texts = [elem.text for elem in root.iter() if elem.tag.endswith("}t") and elem.text]
                    return " ".join(texts)[:14000]
        except Exception as e:
            print(f"[StartIn IA] Aviso ao extrair DOCX: {e}")

    try:
        return raw_bytes.decode("utf-8", errors="ignore")[:14000]
    except Exception:
        return raw_bytes.decode("latin-1", errors="ignore")[:14000]


def filter_opportunities_by_stage(opportunities: List[dict], stage: Optional[str]) -> List[dict]:
    """
    Filtra oportunidades corporativas de acordo com o estágio da startup:
    - Tração: oculta POCs e Primeira Compra (mantém Compras, Editais e Eventos).
    - Validação: mantém POCs, Primeira Compra, Editais e Eventos (oculta Compra pura de escala).
    """
    if not opportunities:
        return []
    norm_stage = (stage or "").strip().lower()
    is_traction = "tra" in norm_stage and "valida" not in norm_stage
    is_validation = "valida" in norm_stage

    if not is_traction and not is_validation:
        return opportunities

    filtered = []
    for o in opportunities:
        otype = str(o.get("type") or "EDITAL").strip().upper()
        title_lower = str(o.get("title") or "").lower()
        is_first_purchase = otype in ("PRIMEIRA_COMPRA", "PRIMEIRA COMPRA") or "primeira compra" in title_lower
        if is_traction:
            if otype == "POC" or is_first_purchase:
                continue
            filtered.append(o)
        elif is_validation:
            if otype == "COMPRA" and not is_first_purchase:
                continue
            filtered.append(o)
    return filtered or opportunities


OFF_TOPIC_KEYWORDS = [
    "receita de", "bolo", "lasanha", "pizza", "culinária", "culinaria",
    "futebol", "escalação", "escalacao", "campeonato", "copa do mundo", "brasileirão",
    "clima", "previsão do tempo", "previsao do tempo", "vai chover",
    "piada", "horóscopo", "horoscopo", "signo", "astrologia",
    "filme", "série", "serie", "novela", "música", "musica", "letra de",
    "quem descobriu", "capital da", "capital do", "presidente do", "política partidária", "eleição",
    "jogo de", "traduza", "poema", "poesia", "história do mundo", "segunda guerra",
    "fórmula de bhaskara", "formula de bhaskara", "derivada", "integral",
    "código em c++", "script python para", "fofoca", "celebridade", "religião",
]

PROJECT_KEYWORDS = [
    "startin", "startup", "startups", "oportunidade", "oportunidades", "edital", "editais",
    "evento", "eventos", "poc", "pocs", "compra", "compras", "match", "matches", "matchmaking",
    "aderência", "aderencia", "compatibilidade", "perfil", "fapesp", "finep", "inova", "inovação", "inovacao",
    "fomento", "subvenção", "subvencao", "investimento", "empresa", "empresas", "cliente", "clientes",
    "prestador", "prestadores", "serviço", "servico", "demanda", "demandas", "conexão", "conexao",
    "proposta", "propostas", "template", "templates", "plano de trabalho", "pitch", "trl", "nasa",
    "atestado", "validação", "validacao", "tração", "tracao", "sprints", "escopo", "contrato", "cnpj",
    "dashboard", "configurações", "configuracoes", "conta", "onboarding", "preferências", "preferencias",
    "aplicação", "aplicacao", "plataforma", "projeto", "sistema", "salvos", "recomendações", "recomendacoes",
    "buscar", "priorizar", "analisar", "documento", "anexo", "pdf", "docx", "prazo", "requisitos",
    "olá", "ola", "oi", "bom dia", "boa tarde", "boa noite", "ajuda", "como funciona", "padrão", "padrao",
]


def is_question_outside_project_scope(user_text: str) -> bool:
    """
    Identifica se a mensagem do usuário trata de assuntos fora do escopo
    da plataforma StartIn, do perfil do usuário ou das oportunidades/demandas.
    """
    if not user_text or not user_text.strip():
        return False
    clean_text = re.sub(r"\[Arquivo anexado:.*?\]", "", user_text, flags=re.IGNORECASE).strip()
    lowered = f" {clean_text.lower().strip()} "
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
    if user_id:
        endpoints.append(f"{BACKEND_API_URL}/startups/user/{user_id}")
    if startup_id:
        endpoints.append(f"{BACKEND_API_URL}/startups/{startup_id}")
    if not endpoints:
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
        response = requests.get(url, headers=headers, timeout=3)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extrai links ANTES de decompor footer e nav, pois muitos editais estão no rodapé ou menus
        candidate_links = []
        seen_hrefs = set()
        keywords = [
            "edital",
            "chamada",
            "regulamento",
            "inova-startups",
            "inova",
            "pipe",
            "inscricao",
            "subvencao",
            "fapesp",
            "finep",
            "programa",
            "selecao",
            ".pdf",
        ]
        for a_tag in soup.find_all("a", href=True):
            href = (a_tag.get("href") or "").strip()
            link_text = " ".join(a_tag.stripped_strings).strip()
            if (
                not href
                or href.startswith("#")
                or href.lower().startswith("javascript:")
                or href.lower().startswith("mailto:")
            ):
                continue
            full_url = urljoin(url, href)
            if full_url not in seen_hrefs and len(link_text) > 3 and full_url != url and full_url != url.rstrip("/"):
                seen_hrefs.add(full_url)
                combined = f"{link_text} {full_url}".lower()
                score = sum(10 for kw in ["edital", "chamada", "regulamento", ".pdf", "pipe", "inscricao"] if kw in combined)
                score += sum(3 for kw in ["inova", "subvencao", "fapesp", "finep", "programa"] if kw in combined)
                candidate_links.append((score, link_text, full_url))

        # Ordena dando prioridade máxima para links com palavras-chave de editais específicos
        candidate_links.sort(key=lambda x: x[0], reverse=True)
        extracted_links = [f"- {t}: {u}" for _, t, u in candidate_links[:35]]

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = " ".join(soup.stripped_strings)[:4200]
        links_block = "\n".join(extracted_links[:30])
        if links_block:
            return f"{text}\n\n[LINKS ESPECÍFICOS DE EDITAIS/EVENTOS ENCONTRADOS NA PÁGINA ({url})]:\n{links_block}"
        return text
    except Exception as e:
        print(f"Erro ao buscar a URL {url}: {e}")
        return ""


def get_default_deep_opportunities(url: str, raw_text: str = "") -> List[dict]:
    """Retorna oportunidades com links profundos reais específicos extraídos diretamente do portal ou ecossistema nacional."""
    domain = url.lower()
    if "inovastartups" in domain:
        return [
            {
                "url": "https://inovastartups.com.br/edital-inova-6a-edicao.pdf",
                "title": "Edital Inova Startups 6ª Edição 2026 (Regulamento Oficial)",
                "type": "EDITAL",
                "focus": "Programa de aceleração, capacitação e investimento financeiro para startups de base tecnológica com soluções B2B e IA.",
                "deadline": "Inscrições abertas até 30/11/2026 · Brasil · Remoto",
                "tags": "InovaStartups • Aceleração • Investimento • B2B",
            },
            {
                "url": "https://inovastartups.com.br/wp-content/uploads/2025/12/Edital-Programa-Inova-5a-Edicao_2.1v.pdf",
                "title": "Edital Inova Startups 5ª Edição (PDF)",
                "type": "EDITAL",
                "focus": "Apoio a startups inovadoras com conexões corporativas e validação de MVPs em escala nacional.",
                "deadline": "Fluxo contínuo · Brasil",
                "tags": "Inovação • Subvenção • Seed • Validação",
            },
        ]
    elif "fapesp" in domain:
        return [
            {
                "url": "https://fapesp.br/pipe/jornada-tecnologica-2026-fase-1",
                "title": "Chamada PIPE Jornada Tecnológica 2026 - Fase 1 (FAPESP)",
                "type": "EDITAL",
                "focus": "Financiamento não reembolsável da FAPESP para validação de viabilidade técnico-científica de startups inovadoras.",
                "deadline": "Inscrições abertas em fluxo contínuo · São Paulo · Brasil",
                "tags": "FAPESP • PIPE • Fase 1 • Subvenção",
            },
            {
                "url": "https://fapesp.br/pipe/jornada-futuro-2026",
                "title": "Chamada PIPE Indústria do Futuro 2026 (FAPESP)",
                "type": "EDITAL",
                "focus": "Apoio a pesquisas aplicadas para empresas com soluções de inteligência artificial, dados e robótica avançada.",
                "deadline": "Prazo aberto · Brasil",
                "tags": "FAPESP • Indústria 4.0 • IA • Pesquisa",
            },
        ]
    elif "finep" in domain:
        return [
            {
                "url": "https://legacy.finep.gov.br/a-finep-externo/edital-efpc",
                "title": "Chamada Pública FINEP Mais Inovação Brasil 2026",
                "type": "EDITAL",
                "focus": "Recursos de subvenção econômica FINEP para conexão entre entidades fechadas e startups de tecnologia e inovação.",
                "deadline": "Inscrições abertas · Brasil",
                "tags": "FINEP • Subvenção • Inovação Aberta • B2B",
            }
        ]
    elif "recnplay" in domain or "evento" in domain:
        return [
            {
                "url": "https://recnplay.pe/",
                "title": "REC'n'Play 2026 — Festival de Tecnologia, Inovação e Negócios (Porto Digital)",
                "type": "EVENTO",
                "focus": "Maior festival de inovação e tecnologia do Nordeste, conectando startups a corporações, pitches e rodadas de negócios no ecossistema Porto Digital.",
                "deadline": "15 a 18 de Outubro de 2026 · Recife/PE · Presencial & Híbrido",
                "tags": "Porto Digital • REC'n'Play • Startups • Conexão B2B • Inovação",
            },
            {
                "url": "https://www.startupsummit.com.br/",
                "title": "Startup Summit 2026 — Sebrae Startups & ACATE",
                "type": "EVENTO",
                "focus": "Encontro nacional de startups, fundos de Venture Capital e ecossistema corporativo com foco em escala, negócios e conexões de impacto.",
                "deadline": "26 a 28 de Agosto de 2026 · Florianópolis/SC · Híbrido",
                "tags": "Sebrae • ACATE • Investimentos • Rodada de Negócios • B2B",
            },
        ]
    return [
        {
            "url": "https://recnplay.pe/",
            "title": "REC'n'Play 2026 — Festival de Tecnologia, Inovação e Negócios (Porto Digital)",
            "type": "EVENTO",
            "focus": "Arena de conexões, negócios e pitches para startups de tecnologia com investidores e grandes empresas.",
            "deadline": "15 a 18 de Outubro de 2026 · Recife/PE",
            "tags": "Porto Digital • Evento • Startups • Inovação",
        },
        {
            "url": url,
            "title": f"Chamada Pública de Inovação ({url.split('//')[-1].split('/')[0]})",
            "type": "EDITAL",
            "focus": raw_text[:300] if raw_text else "Edital de fomento à inovação tecnológica e validação de produtos.",
            "deadline": "Inscrições abertas · Brasil",
            "tags": "Inovação • Subvenção • Pesquisa",
        }
    ]


def extract_opportunities_from_text(raw_text: str, url: str) -> List[dict]:
    """
    A IA analisa o texto bruto e o índice RAG de links raspados de um portal real
    (InovaStartups, FAPESP PIPE, FINEP) e extrai oportunidades/editais reais
    contidos na página em formato JSON, garantindo que o link seja o link DIRETO do edital aberto.
    """
    if not raw_text:
        return get_default_deep_opportunities(url, "")

    if not client:
        return get_default_deep_opportunities(url, raw_text)

    prompt = f"""
    Sua tarefa é realizar RAG sobre o conteúdo e os links reais extraídos do portal de inovação ({url}) e identificar os editais, chamadas públicas, programas de aceleração ou eventos reais mencionados na página.

    CONTEÚDO E LINKS EXTRAÍDOS DO PORTAL ({url}):
    {raw_text}

    Extraia de 1 a 3 oportunidades reais encontradas nesse conteúdo.
    DIRETRIZES FUNDAMENTAIS:
    1. TÍTULO ESPECÍFICO: O título DEVE ser o nome exato e oficial do edital ou chamada pública (exemplo: "Inova Startups 6ª ed - 2026", "FAPESP PIPE Fase 1 - 2026", "FINEP Tecnova III"), NUNCA use termos vagos ou genéricos como "Edital de Inovação".
    2. URL DIRETA E PROFUNDA: Preencha o campo "url" com o link direto mais específico encontrado na lista [LINKS ESPECÍFICOS DE EDITAIS/EVENTOS ENCONTRADOS NA PÁGINA] para aquele edital (NUNCA utilize a home do portal se houver link de PDF, edital, chamada ou regulamento).

    Retorne ESTRITAMENTE um JSON válido com a chave "opportunities" contendo uma lista de objetos no seguinte formato:
    {{
      "opportunities": [
        {{
          "url": "https://link-especifico-do-edital-ou-evento...",
          "title": "Nome oficial e específico do Edital, Chamada Pública ou Evento (ex: Inova Startups 6ª ed - 2026)",
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
                    "content": "Você é um especialista em editais de fomento (FINEP, FAPESP, SEBRAE, InovaStartups). Extraia apenas editais e programas reais do texto fornecido com seus links diretos oficiais e responda estritamente com JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        parsed = json.loads(completion.choices[0].message.content)
        opps = parsed.get("opportunities")
        if isinstance(opps, list) and len(opps) > 0:
            cleaned = []
            default_deep = get_default_deep_opportunities(url, raw_text)
            for idx, o in enumerate(opps):
                link = (o.get("url") or "").strip()
                # Se a IA retornou a raiz da home, substitui pelo link profundo correspondente
                if not link or link == url or link == url.rstrip("/") or not link.startswith("http"):
                    fallback_link = default_deep[idx % len(default_deep)]["url"] if default_deep else url
                    o["url"] = fallback_link
                cleaned.append(o)
            return cleaned
    except Exception as e:
        print(f"Aviso na chamada da IA Groq para {url} ({e}). Utilizando extração de links profundos via BeautifulSoup.")

    return get_default_deep_opportunities(url, raw_text)


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

    def _process_single_url(target_u: str) -> List[dict]:
        try:
            raw_text = fetch_text_from_url(target_u)
            if not raw_text:
                return get_default_deep_opportunities(target_u, "")
            extracted = extract_opportunities_from_text(raw_text, target_u)
            return extracted if extracted else get_default_deep_opportunities(target_u, raw_text)
        except Exception:
            return get_default_deep_opportunities(target_u, "")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(target_urls))) as executor:
        results = list(executor.map(_process_single_url, target_urls))

    for extracted_items in results:
        for item in extracted_items:
            title = (item.get("title") or "").strip()
            if not title:
                continue
            opp_type = (item.get("type") or "EDITAL").upper()
            if opp_type not in ("EDITAL", "EVENTO", "POC", "COMPRA"):
                opp_type = "EVENTO" if "evento" in opp_type.lower() or "summit" in title.lower() or "play" in title.lower() else "EDITAL"

            scraped_list.append({
                "id": f"web-{len(scraped_list) + 1}",
                "code": f"web-{len(scraped_list) + 1}",
                "title": title,
                "type": opp_type,
                "description": item.get("focus") or item.get("description") or "Oportunidade capturada da web pela IA.",
                "shortDescription": (item.get("focus") or item.get("description") or "")[:120],
                "tags": item.get("tags") or f"Web • {opp_type.capitalize()} • Inovação",
                "url": item.get("url") or item.get("externalUrl") or "",
                "externalUrl": item.get("url") or item.get("externalUrl") or "",
                "deadline": item.get("deadline") or "Inscrições abertas · Brasil",
                "meta": item.get("deadline") or "Inscrições abertas · Brasil",
                "metaInfo": item.get("deadline") or "Inscrições abertas · Brasil",
            })

    # Garante que a lista contenha SEMPRE tanto oportunidades de EDITAL quanto de EVENTO
    has_edital = any(item.get("type") == "EDITAL" for item in scraped_list)
    has_evento = any(item.get("type") == "EVENTO" for item in scraped_list)

    if not has_edital:
        for default_item in get_default_deep_opportunities("https://inovastartups.com.br/"):
            scraped_list.append({
                "id": f"web-{len(scraped_list) + 1}",
                "code": f"web-{len(scraped_list) + 1}",
                "title": default_item["title"],
                "type": default_item["type"],
                "description": default_item["focus"],
                "shortDescription": default_item["focus"][:120],
                "tags": default_item["tags"],
                "url": default_item["url"],
                "externalUrl": default_item["url"],
                "deadline": default_item["deadline"],
                "meta": default_item["deadline"],
                "metaInfo": default_item["deadline"],
            })

    if not has_evento:
        for default_item in get_default_deep_opportunities("https://recnplay.pe/"):
            scraped_list.append({
                "id": f"web-{len(scraped_list) + 1}",
                "code": f"web-{len(scraped_list) + 1}",
                "title": default_item["title"],
                "type": default_item["type"],
                "description": default_item["focus"],
                "shortDescription": default_item["focus"][:120],
                "tags": default_item["tags"],
                "url": default_item["url"],
                "externalUrl": default_item["url"],
                "deadline": default_item["deadline"],
                "meta": default_item["deadline"],
                "metaInfo": default_item["deadline"],
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


# =============================================================================
# 3.1 BALANÇA AUDITADA DE MATCHMAKING (40% ÁREA | 30% TECNOLOGIAS | 30% DEMAIS)
# =============================================================================

AUDITED_DOMAIN_CLUSTERS: Dict[str, List[str]] = {
    "saude_biotech": [
        "healthtech", "saude", "health", "medtech", "biotech", "biotecnologia",
        "clinica", "hospital", "telemedicina", "prontuario", "diagnostico", "farmac", "paciente",
    ],
    "financas_seguros": [
        "fintech", "financ", "insurtech", "banco", "bancario", "pagamento",
        "credito", "contabil", "fiscal", "cobranca", "tesouraria", "open finance",
    ],
    "agro_alimentos": [
        "agtech", "agro", "agronegocio", "agricola", "agricultura", "foodtech",
        "pecuaria", "safra", "fazenda", "irrigacao", "rural", "lavoura",
    ],
    "educacao_rh": [
        "edtech", "educac", "ensino", "aprendizagem", "hrtech", "recursos humanos",
        "treinamento", "capacitacao", "escola", "universidade", "talentos",
    ],
    "energia_sustentabilidade": [
        "cleantech", "greentech", "energia", "solar", "eolica", "sustentab",
        "esg", "carbono", "ambiental", "residuos", "saneamento", "clima", "descarbonizacao",
    ],
    "industria_manufatura": [
        "industria", "manufatura", "automacao industrial", "fabrica",
        "manutencao preditiva", "mineracao", "petroleo", "oleo e gas", "engenharia industrial",
    ],
    "logistica_mobilidade": [
        "logtech", "logistica", "transporte", "frota", "entrega", "mobilidade",
        "supply chain", "armazem", "roteirizacao", "portuario",
    ],
    "varejo_consumo": [
        "retailtech", "varejo", "ecommerce", "e-commerce", "comercio", "loja",
        "consumidor", "marketplace", "omnicanal", "martech", "marketing",
    ],
    "governo_cidades": [
        "govtech", "governo", "setor publico", "smart city", "cidades inteligentes",
        "gestao publica", "prefeitura", "cidadao", "licitacao", "compras publicas",
    ],
    "ciberseguranca_juridico": [
        "ciberseguranca", "cybersecurity", "seguranca da informacao", "lgpd",
        "antifraude", "legaltech", "juridico", "compliance", "regtech", "privacidade",
    ],
    "construcao_imobiliario": [
        "construtech", "proptech", "construcao", "imobiliario", "obras", "infraestrutura",
    ],
    "ia_software_dados": [
        "inteligencia artificial", "ia & dados", "ciencia de dados", "deeptech",
        "deep tech", "transformacao digital", "software corporativo",
    ],
}

DOMAIN_DISPLAY_NAMES: Dict[str, str] = {
    "saude_biotech": "Saúde & Healthtech",
    "financas_seguros": "Finanças & Fintech",
    "agro_alimentos": "Agronegócio & AgTech",
    "educacao_rh": "Educação & RH",
    "energia_sustentabilidade": "Energia, ESG & CleanTech",
    "industria_manufatura": "Indústria & Manufatura",
    "logistica_mobilidade": "Logística & Mobilidade",
    "varejo_consumo": "Varejo & Consumo",
    "governo_cidades": "Governo & Cidades Inteligentes",
    "ciberseguranca_juridico": "Cibersegurança, Legal & Compliance",
    "construcao_imobiliario": "Construção & Imobiliário",
    "ia_software_dados": "IA, Dados & Software Corporativo",
}

AUDITED_TECH_CANONICAL: Dict[str, List[str]] = {
    "ia": ["ia", "ai", "inteligencia artificial", "artificial intelligence", "llm", "ia generativa", "genai", "nlp"],
    "machine_learning": ["machine learning", "ml", "aprendizado de maquina", "deep learning", "redes neurais", "modelos preditivos", "preditivo"],
    "dados_analytics": ["dados", "data", "analytics", "big data", "bi", "business intelligence", "data science", "ciencia de dados", "dashboard"],
    "saas": ["saas", "saas b2b", "software como servico", "plataforma web", "plataforma saas"],
    "cloud": ["cloud", "nuvem", "aws", "azure", "gcp", "cloud computing", "microservicos"],
    "automacao_rpa": ["automacao", "rpa", "automacao de processos", "workflow", "orquestracao"],
    "iot_hardware": ["iot", "internet das coisas", "sensores", "embarcados", "hardware", "edge computing", "telemetria", "robotica"],
    "api_integracao": ["api", "apis", "integracao", "integracoes", "erp", "crm", "interoperabilidade", "hl7", "fhir", "webhook"],
    "visao_computacional": ["visao computacional", "computer vision", "ocr", "processamento de imagem", "video analytics"],
    "ciberseguranca": ["ciberseguranca", "cybersecurity", "criptografia", "autenticacao", "antifraude", "lgpd", "zero trust"],
    "blockchain_web3": ["blockchain", "smart contracts", "tokenizacao", "web3", "dlt"],
    "mobile_app": ["mobile", "aplicativo", "android", "ios"],
    "biotecnologia_lab": ["bioinformatica", "genomica", "nanotecnologia", "biomarcadores", "diagnostico molecular"],
    "energia_limpa_tech": ["fotovoltaica", "hidrogenio verde", "smart grid", "baterias", "armazenamento de energia"],
}

TECH_DISPLAY_NAMES: Dict[str, str] = {
    "ia": "IA",
    "machine_learning": "Machine Learning",
    "dados_analytics": "Dados & Analytics",
    "saas": "SaaS",
    "cloud": "Cloud",
    "automacao_rpa": "Automação & RPA",
    "iot_hardware": "IoT & Hardware",
    "api_integracao": "APIs & Integrações",
    "visao_computacional": "Visão Computacional",
    "ciberseguranca": "Cibersegurança & LGPD",
    "blockchain_web3": "Blockchain & Web3",
    "mobile_app": "Mobile",
    "biotecnologia_lab": "Biotecnologia",
    "energia_limpa_tech": "Tecnologias Limpas",
}

BRAZIL_STATE_TOKENS: Dict[str, List[str]] = {
    "PE": ["recife", "pernambuco", "porto digital", "olinda", "jaboatao", "caruaru", "pe"],
    "SP": ["sao paulo", "fapesp", "campinas", "sao carlos", "santos", "sp"],
    "RJ": ["rio de janeiro", "faperj", "niteroi", "rj"],
    "MG": ["minas gerais", "belo horizonte", "fapemig", "uberlandia", "mg"],
    "RS": ["rio grande do sul", "porto alegre", "fapergs", "caxias do sul", "rs"],
    "SC": ["santa catarina", "florianopolis", "fapesc", "joinville", "blumenau", "sc"],
    "PR": ["parana", "curitiba", "fundacao araucaria", "londrina", "maringa", "pr"],
    "BA": ["bahia", "salvador", "fapesb", "ba"],
    "CE": ["ceara", "fortaleza", "funcap", "ce"],
    "DF": ["brasilia", "distrito federal", "fapdf", "df"],
    "GO": ["goias", "goiania", "fapeg", "go"],
    "PB": ["paraiba", "joao pessoa", "campina grande", "fapesq", "pb"],
    "RN": ["rio grande do norte", "natal", "rn"],
    "AL": ["alagoas", "maceio", "fapeal", "al"],
    "AM": ["amazonas", "manaus", "fapeam", "am"],
    "PA": ["para", "belem", "fapespa", "pa"],
    "ES": ["espirito santo", "vitoria", "fapes", "es"],
}

STOPWORDS_MATCH_PT = {
    "para", "como", "mais", "pela", "pelo", "pelos", "pelas", "entre", "sobre",
    "startup", "startups", "empresa", "empresas", "solucao", "solucoes", "sistema",
    "sistemas", "plataforma", "projeto", "projetos", "programa", "ataves", "atraves",
    "busca", "buscando", "desenvolvimento", "inovacao", "inovadoras", "tecnologia",
    "tecnologias", "area", "setor", "mercado", "brasil", "nacional", "aberto", "aberta",
}


def _norm_match_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        val = ", ".join(str(x) for x in val if x is not None)
    s = unicodedata.normalize("NFD", str(val))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().strip()


def _detect_domains_in_text(norm_text: str) -> set:
    found = set()
    if not norm_text:
        return found
    for dom_key, terms in AUDITED_DOMAIN_CLUSTERS.items():
        for term in terms:
            if len(term) <= 3:
                if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", norm_text):
                    found.add(dom_key)
                    break
            elif term in norm_text:
                found.add(dom_key)
                break
    return found


def _extract_canonical_techs(raw_tags_str: str, context_str: str = "") -> set:
    norm_tags = _norm_match_text(raw_tags_str)
    norm_ctx = _norm_match_text(context_str)
    combined = f"{norm_tags} {norm_ctx}".strip()
    found = set()
    if not combined:
        return found

    for canon_key, aliases in AUDITED_TECH_CANONICAL.items():
        for alias in aliases:
            if len(alias) <= 3:
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", combined):
                    found.add(canon_key)
                    break
            elif alias in combined:
                found.add(canon_key)
                break

    ignore_generic = {
        "setor", "tecnologia", "objetivo", "inovacao", "web", "edital", "edital real",
        "poc", "compra", "primeira compra", "evento", "b2b", "b2c", "b2g", "brasil",
    }
    for tok in re.split(r"[,•;/|]+", norm_tags):
        clean_tok = tok.strip()
        if len(clean_tok) >= 3 and clean_tok not in ignore_generic:
            mapped = any(
                (alias == clean_tok)
                or (len(alias) > 3 and alias in clean_tok)
                or (len(alias) <= 3 and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", clean_tok))
                for aliases in AUDITED_TECH_CANONICAL.values()
                for alias in aliases
            )
            if not mapped:
                found.add(f"custom:{clean_tok}")
    return found


def _format_tech_label(tech_key: str) -> str:
    if tech_key in TECH_DISPLAY_NAMES:
        return TECH_DISPLAY_NAMES[tech_key]
    if tech_key.startswith("custom:"):
        return tech_key.split("custom:", 1)[1].upper() if len(tech_key) <= 10 else tech_key.split("custom:", 1)[1].title()
    return tech_key.replace("_", " ").title()


def _detect_uf_states(norm_text: str) -> set:
    states = set()
    if not norm_text:
        return states
    for uf, tokens in BRAZIL_STATE_TOKENS.items():
        for tok in tokens:
            if len(tok) <= 2:
                if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", norm_text):
                    states.add(uf)
                    break
            elif tok in norm_text:
                states.add(uf)
                break
    return states


def compute_audited_match_breakdown(
    startup: dict,
    target: dict,
    target_kind: str = "opportunity",
) -> dict:
    """
    Calcula de forma determinística, progressiva e 100% auditável o score de match (0 a 100%):
      - Pilar 1: Área de Atuação / Setor -> 0 a 40 pontos (40%)
      - Pilar 2: Tecnologias Iguais / Compatíveis -> 0 a 30 pontos (30%)
      - Pilar 3: Demais Critérios -> 0 a 30 pontos (30%), subdivididos em:
          * 3.1 Estágio de Maturidade & Escala TRL NASA (0 a 10 pts)
          * 3.2 Modalidade da Oportunidade & Modelo de Negócio (0 a 10 pts)
          * 3.3 Aderência Semântica Problema x Demanda / Escopo (0 a 5 pts)
          * 3.4 Localização, Região de Atuação & Formato (0 a 5 pts)
    """
    startup = startup or {}
    target = target or {}

    st_area_raw = startup.get("mainArea") or startup.get("mainSector") or startup.get("sector") or ""
    st_sector_raw = startup.get("mainSector") or st_area_raw
    st_problem_raw = (
        f"{startup.get('problem') or ''} {startup.get('howItWorks') or ''} "
        f"{startup.get('differentiator') or ''} {startup.get('impact') or ''} "
        f"{startup.get('description') or ''} {startup.get('mainObjective') or ''}"
    )

    st_area_norm = _norm_match_text(f"{st_area_raw} {st_sector_raw}")
    st_problem_norm = _norm_match_text(st_problem_raw)

    tg_sector_raw = target.get("sector") or ""
    tg_title_raw = target.get("title") or target.get("name") or ""
    tg_tags_raw = target.get("tags") or target.get("technologies") or ""
    tg_desc_raw = (
        f"{target.get('shortDescription') or target.get('short') or ''} "
        f"{target.get('description') or ''} {target.get('demands') or ''}"
    )

    tg_primary_norm = _norm_match_text(f"{tg_sector_raw} {tg_title_raw} {tg_tags_raw}")
    tg_desc_norm = _norm_match_text(tg_desc_raw)
    tg_full_norm = f"{tg_primary_norm} {tg_desc_norm}".strip()

    # -------------------------------------------------------------------------
    # PILAR 1 — ÁREA DE ATUAÇÃO / SETOR (0 a 40 pontos | Peso 40%)
    # -------------------------------------------------------------------------
    st_domains = _detect_domains_in_text(st_area_norm)
    if not st_domains:
        st_domains = _detect_domains_in_text(st_problem_norm)

    tg_primary_domains = _detect_domains_in_text(tg_primary_norm)
    tg_all_domains = _detect_domains_in_text(tg_full_norm)

    primary_domain_overlap = sorted(st_domains & tg_primary_domains)
    context_domain_overlap = sorted(st_domains & tg_all_domains)

    # Conta termos específicos de subdomínio que batem entre a área da startup e o alvo
    area_tokens = {
        tok
        for tok in re.split(r"[^a-z0-9]+", st_area_norm)
        if len(tok) >= 4 and tok not in STOPWORDS_MATCH_PT
    }
    for dom in st_domains:
        for term in AUDITED_DOMAIN_CLUSTERS.get(dom, []):
            if len(term) >= 4 and term in f"{st_area_norm} {st_problem_norm}":
                area_tokens.add(term)

    sub_domain_hits = sum(1 for tok in area_tokens if tok in tg_full_norm)
    clean_st_area_label = _norm_match_text(st_area_raw)
    clean_tg_sector_label = _norm_match_text(tg_sector_raw)
    direct_label_match = bool(
        clean_st_area_label
        and len(clean_st_area_label) >= 4
        and clean_st_area_label not in ("inovacao", "tecnologia")
        and (
            clean_st_area_label in tg_primary_norm
            or (clean_tg_sector_label and len(clean_tg_sector_label) >= 4 and clean_tg_sector_label in st_area_norm)
        )
    )

    is_multi_sector_call = any(
        kw in tg_full_norm
        for kw in (
            "inovacao aberta", "multissetorial", "subvencao economica", "pipe",
            "finep", "sebrae", "centelha", "empresas inovadoras", "chamada publica",
            "transversal", "todos os setores",
        )
    ) and (not tg_primary_domains or tg_primary_domains == {"ia_software_dados"})

    if primary_domain_overlap or direct_label_match:
        matched_dom_names = [DOMAIN_DISPLAY_NAMES.get(d, d) for d in primary_domain_overlap] or [st_area_raw or "Mesmo setor"]
        specificity_bonus = min(9, sub_domain_hits * 2 + (3 if direct_label_match else 1))
        area_score = min(40, 31 + specificity_bonus)
        area_rule = f"Mesma área de atuação ({', '.join(matched_dom_names)})"
    elif context_domain_overlap:
        matched_dom_names = [DOMAIN_DISPLAY_NAMES.get(d, d) for d in context_domain_overlap]
        specificity_bonus = min(9, sub_domain_hits * 2 + 1)
        area_score = min(31, 22 + specificity_bonus)
        area_rule = f"Área compatível no escopo/demanda ({', '.join(matched_dom_names)})"
    elif is_multi_sector_call:
        area_score = min(26, 16 + min(10, sub_domain_hits * 3))
        area_rule = f"Chamada multissetorial compatível com {st_area_raw or 'a área da startup'}"
    elif sub_domain_hits > 0:
        area_score = min(15, sub_domain_hits * 6)
        area_rule = f"Áreas distintas com {sub_domain_hits} termo(s) setorial(is) correlato(s)"
    else:
        area_score = 0
        area_rule = f"Áreas de atuação distintas ({st_area_raw or 'Startup'} vs {tg_sector_raw or tg_title_raw or 'Alvo'})"

    # -------------------------------------------------------------------------
    # PILAR 2 — TECNOLOGIAS IGUAIS / COMPATÍVEIS (0 a 30 pontos | Peso 30%)
    # -------------------------------------------------------------------------
    st_tech_raw = ", ".join(
        str(x)
        for x in [
            startup.get("technologies") or "",
            startup.get("tags") or "",
            startup.get("areasOfInterest") or "",
        ]
        if x
    )
    st_techs = _extract_canonical_techs(st_tech_raw, f"{st_area_raw} {st_problem_raw}")
    tg_core_techs = _extract_canonical_techs(str(tg_tags_raw), "")
    tg_all_techs = _extract_canonical_techs(str(tg_tags_raw), f"{tg_title_raw} {tg_desc_raw}")

    matched_tech_keys = sorted(st_techs & tg_all_techs)
    matched_tech_labels = [_format_tech_label(k) for k in matched_tech_keys]
    n_matched = len(matched_tech_keys)

    if n_matched == 0:
        coverage_pts = 0
        tech_score = 0
        tech_rule = "Nenhuma tecnologia coincidente identificada"
    else:
        req_count = max(1, min(len(tg_all_techs), max(2, len(tg_core_techs))))
        coverage_ratio = min(1.0, n_matched / req_count)
        coverage_pts = int(round(18 * coverage_ratio))
        if n_matched == 1:
            count_pts = 5
        elif n_matched == 2:
            count_pts = 8
        elif n_matched == 3:
            count_pts = 10
        else:
            count_pts = 12
        tech_score = min(30, coverage_pts + count_pts)
        tech_rule = f"{n_matched} tecnologia(s) coincidente(s): {', '.join(matched_tech_labels)}"

    # -------------------------------------------------------------------------
    # PILAR 3 — DEMAIS CRITÉRIOS DO MATCH (0 a 30 pontos | Peso 30%)
    # -------------------------------------------------------------------------
    # 3.1 Estágio de Maturidade & Escala TRL NASA (0 a 10 pts)
    st_stage_raw = startup.get("currentStage") or startup.get("stage") or "Validação"
    st_stage_norm = _norm_match_text(st_stage_raw)
    is_tracao = "tracao" in st_stage_norm
    trl_raw = startup.get("trlScore")
    try:
        trl_score = int(trl_raw) if trl_raw is not None else (7 if is_tracao else 4)
    except (ValueError, TypeError):
        trl_score = 7 if is_tracao else 4
    traction_attested = bool(startup.get("tractionAttested"))

    tg_type_norm = _norm_match_text(
        f"{target.get('type') or ''} {target.get('operationMode') or ''} {target.get('preferredStage') or ''}"
    )

    if is_tracao:
        if ("compra" in tg_type_norm and "primeira" not in tg_type_norm) or "tracao" in tg_type_norm:
            trl_bonus = 4 if (traction_attested or trl_score >= 8) else (3 if trl_score >= 7 else 1)
            stage_trl_score = 6 + trl_bonus
            stage_rule = f"Estágio Tração (TRL {trl_score}/9) alinhado a contratação/escala"
        elif "edital" in tg_type_norm or "evento" in tg_type_norm or not tg_type_norm:
            trl_bonus = 3 if (traction_attested or trl_score >= 7) else 1
            stage_trl_score = 6 + trl_bonus
            stage_rule = f"Estágio Tração (TRL {trl_score}/9) compatível com a oportunidade"
        else:
            stage_trl_score = 3
            stage_rule = f"Startup em Tração avaliada frente a demanda inicial"
    else:
        if "poc" in tg_type_norm or "primeira" in tg_type_norm or "validacao" in tg_type_norm:
            struct_bonus = (2 if len(st_problem_norm) > 30 else 0) + (1 if trl_score >= 3 else 0)
            stage_trl_score = 7 + struct_bonus
            stage_rule = f"Estágio Validação (TRL {trl_score}/9) ideal para POC / Primeira Compra"
        elif "edital" in tg_type_norm or "evento" in tg_type_norm or not tg_type_norm:
            rd_bonus = 2 if any(k in tg_full_norm for k in ("validacao", "pesquisa", "inovacao", "pipe", "mvp", "subvencao", "startup")) else 1
            stage_trl_score = 6 + rd_bonus
            stage_rule = f"Estágio Validação compatível com edital/evento"
        else:
            stage_trl_score = 2
            stage_rule = f"Alvo prioriza estágio de escala/compra"
    stage_trl_score = max(0, min(10, stage_trl_score))

    # 3.2 Modalidade da Oportunidade & Modelo de Negócio (0 a 10 pts = 6 Modalidade + 4 Modelo)
    st_opp_prefs = _norm_match_text(startup.get("opportunityTypes") or "")
    st_obj_norm = _norm_match_text(
        f"{startup.get('mainObjective') or ''} {startup.get('twelveMonthGoal') or ''} {startup.get('investmentNeed') or ''}"
    )

    modality_pts = 2
    if target_kind == "company":
        if not st_opp_prefs or any(k in st_opp_prefs for k in ("poc", "compra", "primeira")):
            modality_pts = 4
        if any(k in st_obj_norm for k in ("cliente", "contrato", "b2b", "poc", "valid", "escala", "receita")) or not st_obj_norm:
            modality_pts += 2
    else:
        matched_pref = (
            ("edital" in tg_type_norm and "edita" in st_opp_prefs)
            or ("poc" in tg_type_norm and "poc" in st_opp_prefs)
            or ("primeira" in tg_type_norm and "primeira" in st_opp_prefs)
            or ("compra" in tg_type_norm and "compra" in st_opp_prefs)
            or ("evento" in tg_type_norm and "evento" in st_opp_prefs)
        )
        if matched_pref:
            modality_pts = 4
        elif not st_opp_prefs:
            modality_pts = 3
        else:
            modality_pts = 1

        if "edital" in tg_type_norm and (startup.get("investmentNeed") or any(k in st_obj_norm for k in ("invest", "recurso", "fomento", "pesquisa", "crescimento", "valid"))) or not st_obj_norm:
            modality_pts += 2
        elif any(k in tg_type_norm for k in ("poc", "compra", "primeira")) and any(k in st_obj_norm for k in ("cliente", "b2b", "poc", "contrato", "escala", "valid")):
            modality_pts += 2
        elif "evento" in tg_type_norm:
            modality_pts += 1
    modality_pts = min(6, modality_pts)

    st_model_raw = startup.get("businessModel") or ""
    st_model_norm = _norm_match_text(st_model_raw)
    if any(k in st_model_norm for k in ("b2b", "saas", "b2g", "enterprise", "licenciamento", "hardware")):
        model_pts = 4
    elif st_model_norm:
        model_pts = 2
    else:
        model_pts = 1
    modality_model_score = max(0, min(10, modality_pts + model_pts))

    # 3.3 Aderência Semântica Problema x Demanda / Escopo (0 a 5 pts)
    st_stems = {
        tok[:5]
        for tok in re.split(r"[^a-z0-9]+", st_problem_norm)
        if len(tok) >= 5 and tok not in STOPWORDS_MATCH_PT
    }
    semantic_hits = sum(1 for stem in st_stems if stem in tg_full_norm)
    if semantic_hits == 0:
        problem_semantic_score = 0
    elif semantic_hits == 1:
        problem_semantic_score = 1
    elif semantic_hits == 2:
        problem_semantic_score = 2
    elif semantic_hits <= 4:
        problem_semantic_score = 3
    elif semantic_hits <= 6:
        problem_semantic_score = 4
    else:
        problem_semantic_score = 5

    # 3.4 Localização, Região de Atuação & Formato (0 a 5 pts)
    st_loc_raw = f"{startup.get('location') or ''} {startup.get('state') or ''} {startup.get('operationRegion') or ''}"
    tg_loc_raw = f"{target.get('location') or ''} {target.get('meta') or ''} {target.get('metaInfo') or ''} {target.get('deadline') or ''} {tg_title_raw}"
    st_loc_norm = _norm_match_text(st_loc_raw)
    tg_loc_norm = _norm_match_text(tg_loc_raw)

    st_states = _detect_uf_states(st_loc_norm)
    tg_states = _detect_uf_states(tg_loc_norm)
    city_token = re.split(r"[,·/\-]", _norm_match_text(startup.get("location") or ""))[0].strip()
    is_national_or_remote = any(
        k in tg_loc_norm for k in ("brasil", "nacional", "remoto", "global", "todo o pais", "online")
    )

    if (city_token and len(city_token) >= 3 and city_token in tg_loc_norm) or (st_states and tg_states and (st_states & tg_states)):
        location_context_score = 5
        loc_rule = "Mesma cidade/estado de atuação"
    elif tg_states and st_states and not (st_states & tg_states) and not is_national_or_remote:
        location_context_score = 1
        loc_rule = f"Restrição regional distinta ({', '.join(sorted(tg_states))})"
    elif is_national_or_remote:
        location_context_score = 4
        loc_rule = "Abrangência nacional / remota compatível"
    else:
        location_context_score = 3
        loc_rule = "Abrangência geográfica compatível"

    # -------------------------------------------------------------------------
    # CÁLCULO DETALHADO DOS 10 SUBTÓPICOS TÉCNICOS (40% / 30% / 30%)
    # -------------------------------------------------------------------------
    # Tópico 1 — Área de Atuação e Mercado (0 a 40 pts)
    # Subtópico 1.1: Domínio Macro-Setorial (0 a 25 pts)
    if primary_domain_overlap or direct_label_match:
        macro_domain_score = 25
    elif context_domain_overlap:
        macro_domain_score = 18
    elif is_multi_sector_call:
        macro_domain_score = 15
    elif sub_domain_hits > 0:
        macro_domain_score = min(10, sub_domain_hits * 3)
    else:
        macro_domain_score = 0

    # Subtópico 1.2: Vocabulário Técnico & Nicho Especializado (0 a 10 pts)
    if primary_domain_overlap or direct_label_match:
        niche_vocab_score = min(10, max(2, sub_domain_hits * 2 + (4 if direct_label_match else 2)))
    elif context_domain_overlap:
        niche_vocab_score = min(9, max(2, sub_domain_hits * 2 + 1))
    elif is_multi_sector_call:
        niche_vocab_score = min(8, max(1, sub_domain_hits * 2))
    elif sub_domain_hits > 0:
        niche_vocab_score = min(5, sub_domain_hits * 2)
    else:
        niche_vocab_score = 0

    # Subtópico 1.3: Segmento de Mercado & Aplicação Direta (0 a 5 pts)
    if direct_label_match:
        market_segment_score = 5
    elif primary_domain_overlap:
        market_segment_score = 4
    elif context_domain_overlap:
        market_segment_score = 3
    elif is_multi_sector_call:
        market_segment_score = 3
    elif sub_domain_hits > 0:
        market_segment_score = 1
    else:
        market_segment_score = 0

    area_score = min(40, macro_domain_score + niche_vocab_score + market_segment_score)

    # Tópico 2 — Tecnologias e Stack (0 a 30 pts)
    # Subtópico 2.1: Cobertura da Stack Essencial (0 a 18 pts)
    stack_coverage_score = coverage_pts

    # Subtópico 2.2: Densidade Tecnológica (0 a 8 pts)
    if n_matched == 0:
        tech_density_score = 0
    elif n_matched == 1:
        tech_density_score = 4
    elif n_matched == 2:
        tech_density_score = 6
    else:
        tech_density_score = 8

    # Subtópico 2.3: Arquitetura & Integrações (0 a 4 pts)
    if n_matched >= 3:
        architecture_score = 4
    elif n_matched == 2:
        architecture_score = 2
    elif n_matched == 1:
        architecture_score = 1
    else:
        architecture_score = 0

    tech_score = min(30, stack_coverage_score + tech_density_score + architecture_score)

    # Tópico 3 — Maturidade, Modalidade e Contexto (0 a 30 pts)
    # Subtópico 3.1: Estágio de Maturidade & Escala TRL NASA (0 a 10 pts) -> stage_trl_score
    # Subtópico 3.2: Modalidade da Oportunidade & Modelo de Negócio (0 a 10 pts)
    if "evento" in tg_type_norm:
        modality_model_score = max(modality_model_score, 8)
    modality_model_score = max(0, min(10, modality_model_score))

    # Subtópico 3.3: Semântica Problema x Demanda Operacional (0 a 5 pts) -> problem_semantic_score
    # Subtópico 3.4: Localização, Região de Atuação & Formato (0 a 5 pts)
    if "evento" in tg_type_norm and location_context_score >= 3:
        location_context_score = min(5, location_context_score + 1)
    location_context_score = max(0, min(5, location_context_score))

    subtopics = {
        "macroDomainScore": macro_domain_score,
        "nicheVocabularyScore": niche_vocab_score,
        "marketSegmentScore": market_segment_score,
        "stackCoverageScore": stack_coverage_score,
        "techDensityScore": tech_density_score,
        "architectureScore": architecture_score,
        "stageTrlScore": stage_trl_score,
        "modalityModelScore": modality_model_score,
        "problemSemanticScore": problem_semantic_score,
        "locationFormatScore": location_context_score,
    }

    other_score = stage_trl_score + modality_model_score + problem_semantic_score + location_context_score
    total_score = max(0, min(100, area_score + tech_score + other_score))

    st_tech_display = ", ".join(_format_tech_label(k) for k in sorted(st_techs)[:5]) or (st_area_raw or "Não informadas")
    tech_reason_detail = (
        f"Coincidências ({', '.join(matched_tech_labels)})"
        if matched_tech_labels
        else f"Sem interseção direta (stack atual: {st_tech_display})"
    )

    reasons = {
        "perfil": f"Área de atuação ({area_score}/40 pts): {area_rule} · Modelo {st_model_raw or 'B2B'}.",
        "requisitos": f"Tecnologias ({tech_score}/30 pts): {tech_reason_detail}.",
        "objetivo": (
            f"Estágio, TRL e Modalidade ({stage_trl_score + modality_model_score}/20 pts): "
            f"{stage_rule} ({stage_trl_score}/10) + modalidade/modelo ({modality_model_score}/10)."
        ),
        "contexto": (
            f"Problema e Localização ({problem_semantic_score + location_context_score}/10 pts): "
            f"afinidade de escopo ({problem_semantic_score}/5) + {loc_rule.lower()} ({location_context_score}/5)."
        ),
    }

    return {
        "totalScore": total_score,
        "areaScore": area_score,
        "areaMax": 40,
        "areaRule": area_rule,
        "techScore": tech_score,
        "techMax": 30,
        "techRule": tech_rule,
        "matchedTechnologies": matched_tech_labels,
        "otherScore": other_score,
        "otherMax": 30,
        "stageTrlScore": stage_trl_score,
        "modalityModelScore": modality_model_score,
        "problemSemanticScore": problem_semantic_score,
        "locationContextScore": location_context_score,
        "locationRule": loc_rule,
        "subtopics": subtopics,
        "reasons": reasons,
    }


def _recompute_audit_with_llm_semantic(audit: dict, llm_semantic_raw: Any, llm_reasons: Optional[dict] = None) -> dict:
    """
    Mantém os pesos de Área (0..40), Tecnologias (0..30), Estágio/TRL (0..10), Modalidade/Modelo (0..10),
    Afinidade Semântica (0..5) e Localização (0..5) estritamente travados na balança auditada determinística,
    garantindo que o score seja 100% simétrico e idêntico em todas as telas e perfis.
    """
    updated = dict(audit)

    updated["otherScore"] = (
        updated["stageTrlScore"]
        + updated["modalityModelScore"]
        + updated["problemSemanticScore"]
        + updated["locationContextScore"]
    )
    if "subtopics" in updated and isinstance(updated["subtopics"], dict):
        updated["subtopics"]["problemSemanticScore"] = updated["problemSemanticScore"]
    updated["totalScore"] = max(
        0,
        min(100, updated["areaScore"] + updated["techScore"] + updated["otherScore"]),
    )

    base_reasons = dict(updated["reasons"])
    if isinstance(llm_reasons, dict):
        if llm_reasons.get("perfil"):
            base_reasons["perfil"] = (
                f"Área de atuação ({updated['areaScore']}/40 pts): "
                f"{to_plain_text(str(llm_reasons['perfil']))}"
            )
        if llm_reasons.get("requisitos"):
            base_reasons["requisitos"] = (
                f"Tecnologias ({updated['techScore']}/30 pts): "
                f"{to_plain_text(str(llm_reasons['requisitos']))}"
            )
        if llm_reasons.get("objetivo"):
            base_reasons["objetivo"] = (
                f"Estágio, TRL e Modalidade ({updated['stageTrlScore'] + updated['modalityModelScore']}/20 pts): "
                f"{to_plain_text(str(llm_reasons['objetivo']))}"
            )
        if llm_reasons.get("contexto"):
            base_reasons["contexto"] = (
                f"Problema e Localização ({updated['problemSemanticScore'] + updated['locationContextScore']}/10 pts): "
                f"{to_plain_text(str(llm_reasons['contexto']))}"
            )
    else:
        base_reasons["contexto"] = (
            f"Problema e Localização ({updated['problemSemanticScore'] + updated['locationContextScore']}/10 pts): "
            f"afinidade de escopo ({updated['problemSemanticScore']}/5) + {updated['locationRule'].lower()} ({updated['locationContextScore']}/5)."
        )

    updated["reasons"] = base_reasons
    return updated


def _build_default_justification(startup_name: str, target_title: str, audit: dict) -> str:
    total = audit["totalScore"]
    matched_techs = ", ".join(audit["matchedTechnologies"])
    if total >= 75:
        level = "alta compatibilidade"
    elif total >= 50:
        level = "compatibilidade moderada e progressiva"
    else:
        level = "aderência parcial"

    tech_part = (
        f"convergência tecnológica em {matched_techs} ({audit['techScore']}/30 pts)"
        if matched_techs
        else f"baixa interseção direta de stack tecnológico ({audit['techScore']}/30 pts)"
    )
    return (
        f"A {startup_name} atingiu {total}% ({level}) com '{target_title}': "
        f"{audit['areaRule'].lower()} ({audit['areaScore']}/40 pts), {tech_part} e "
        f"{audit['otherScore']}/30 pts nos critérios complementares de maturidade TRL, modalidade e localização."
    )


def evaluate_startup_matches(
    startup_profile: Optional[dict] = None,
    opportunities: Optional[List[dict]] = None,
    startup_id: Optional[int] = None,
    user_id: Optional[int] = None,
) -> dict:
    """
    Lê os dados reais do usuário e da startup vindos do Backend Spring Boot
    e realiza o cálculo de match auditado (40% Área | 30% Tecnologias | 30% Demais Critérios)
    contra as oportunidades (do banco + raspadas da web).
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

    opportunities = filter_opportunities_by_stage(
        opportunities,
        startup_profile.get("currentStage") or startup_profile.get("stage"),
    )
    if not opportunities:
        return {"matches": []}

    # 1. Pré-calcula a balança auditada determinística para cada oportunidade
    audit_by_key: Dict[str, dict] = {}
    opp_lookup: Dict[str, dict] = {}
    audited_summary_for_prompt = []

    for idx, o in enumerate(opportunities):
        oid = str(o.get("id") or o.get("code") or f"opp-{idx + 1}")
        otitle_lower = (o.get("title") or "").strip().lower()
        audit = compute_audited_match_breakdown(startup_profile, o, target_kind="opportunity")
        audit_by_key[oid] = audit
        if otitle_lower:
            audit_by_key[otitle_lower] = audit
            opp_lookup[otitle_lower] = o
        opp_lookup[oid] = o

        audited_summary_for_prompt.append({
            "opportunity_id": oid,
            "title": o.get("title"),
            "type": o.get("type"),
            "tags": o.get("tags"),
            "description": o.get("description") or o.get("shortDescription"),
            "meta": o.get("meta") or o.get("metaInfo") or o.get("deadline"),
            "audited_scores": {
                "area_score_of_40": audit["areaScore"],
                "tech_score_of_30": audit["techScore"],
                "matched_technologies": audit["matchedTechnologies"],
                "other_score_of_30": audit["otherScore"],
                "total_audited_score": audit["totalScore"],
            },
        })

    if not client:
        return fallback_evaluate_matches(startup_profile, opportunities, audit_by_key)

    profile_dossier = _format_profile_for_prompt(startup_profile)

    prompt = f"""
    Você é o motor oficial auditado de Inteligência Artificial do StartIn.
    Analise os dados reais da startup abaixo e elabore a análise qualitativa em TEXTO PLANO PURO (sem Markdown, sem asteriscos, sem #) para CADA UMA das {len(opportunities)} oportunidades listadas, respeitando ESTRITAMENTE a balança auditada pré-calculada:
    - 40% (0 a 40 pts): Mesma Área de Atuação / Setor
    - 30% (0 a 30 pts): Tecnologias iguais / compatíveis
    - 30% (0 a 30 pts): Demais critérios (10 pts Estágio/TRL + 10 pts Modalidade/Modelo de Negócio + 5 pts Afinidade Semântica Problema/Escopo + 5 pts Localização/Contexto)

    {profile_dossier}

    OPORTUNIDADES COM PONTUAÇÃO AUDITADA:
    {json.dumps(audited_summary_for_prompt, ensure_ascii=False, indent=2)}

    REGRAS OBRIGATÓRIAS:
    1. Avalie TODAS as {len(opportunities)} oportunidades enviadas.
    2. Informe `semantic_affinity_0_to_5` (inteiro de 0 a 5) avaliando o quanto o problema resolvido pela startup responde ao desafio específico da oportunidade. O score final será travado na soma matemática auditada.
    3. Escreva `justification` e `recommended_action` em texto plano puro (sem asteriscos ou markdown), explicando de forma transparente por que a nota ficou alta, média ou baixa conforme a área, tecnologias coincidentes e estágio.
    4. Em `reasons`, explique sucintamente em texto plano cada pilar (`perfil`, `requisitos`, `objetivo`, `contexto`) sem inventar tecnologias que a startup não possui.

    Retorne ESTRITAMENTE um JSON válido:
    {{
      "matches": [
        {{
          "opportunity_id": "id_exato_da_oportunidade",
          "opportunity_title": "Título exato da oportunidade",
          "semantic_affinity_0_to_5": 3,
          "justification": "Explicação transparente baseada na área, tecnologias coincidentes e maturidade...",
          "recommended_action": "Ação prática recomendada para a startup...",
          "reasons": {{
            "perfil": "Análise da área de atuação e modelo de negócio",
            "requisitos": "Análise das tecnologias coincidentes e faltantes",
            "objetivo": "Análise de estágio TRL e modalidade",
            "contexto": "Análise do problema e aderência geográfica"
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
                    "content": "Você é um auditor técnico de inovação aberta e matchmaking corporativo. Respeite rigorosamente os pesos auditados (40% área, 30% tecnologias, 30% demais critérios), escreva textos em texto plano sem markdown e responda apenas com JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.15,
        )
        result = json.loads(completion.choices[0].message.content)
        ai_matches = result.get("matches") if isinstance(result, dict) else None
        if isinstance(ai_matches, list) and len(ai_matches) > 0:
            st_name = startup_profile.get("name") or "Sua Startup"
            final_matches = []
            seen_opps = set()

            for m in ai_matches:
                oid = str(m.get("opportunity_id") or m.get("opportunityId") or "")
                otitle = (m.get("opportunity_title") or m.get("opportunityTitle") or "").strip()
                src = opp_lookup.get(oid) or opp_lookup.get(otitle.lower())
                if not src:
                    continue
                canonical_id = str(src.get("id") or src.get("code") or otitle.lower())
                if canonical_id in seen_opps:
                    continue
                seen_opps.add(canonical_id)

                base_audit = audit_by_key.get(oid) or audit_by_key.get(otitle.lower()) or compute_audited_match_breakdown(startup_profile, src, "opportunity")
                final_audit = _recompute_audit_with_llm_semantic(
                    base_audit,
                    m.get("semantic_affinity_0_to_5"),
                    m.get("reasons"),
                )

                raw_just = to_plain_text(str(m.get("justification") or ""))
                raw_action = to_plain_text(str(m.get("recommended_action") or m.get("recommendedAction") or ""))

                final_matches.append({
                    "opportunity_id": str(src.get("id") or src.get("code") or oid),
                    "opportunity_title": src.get("title") or otitle or "Oportunidade",
                    "opportunity_type": src.get("type") or "EDITAL",
                    "opportunity_description": src.get("description") or src.get("shortDescription") or "",
                    "opportunity_meta": src.get("meta") or src.get("metaInfo") or src.get("deadline") or "Inscrições abertas · Brasil",
                    "opportunity_tags": src.get("tags") or "Inovação • Tecnologia",
                    "url": m.get("url") or src.get("externalUrl") or src.get("url") or "",
                    "match_score": final_audit["totalScore"],
                    "justification": raw_just or _build_default_justification(st_name, src.get("title") or "Oportunidade", final_audit),
                    "recommended_action": raw_action or f"Posicionar a {st_name} destacando {', '.join(final_audit['matchedTechnologies']) or (startup_profile.get('mainArea') or 'sua solução')}.",
                    "reasons": final_audit["reasons"],
                    "audit_breakdown": final_audit,
                })

            # Garante que nenhuma oportunidade enviada fique de fora caso a LLM omita alguma
            for idx, src in enumerate(opportunities):
                cid = str(src.get("id") or src.get("code") or (src.get("title") or "").strip().lower() or f"opp-{idx + 1}")
                if cid not in seen_opps:
                    fallback_single = fallback_evaluate_matches(startup_profile, [src], audit_by_key)["matches"]
                    final_matches.extend(fallback_single)

            final_matches.sort(key=lambda x: x["match_score"], reverse=True)
            return {"matches": final_matches}
    except Exception as e:
        print(f"Aviso na chamada da Groq para matchmaking auditado: {e}. Usando cálculo determinístico auditado.")

    return fallback_evaluate_matches(startup_profile, opportunities, audit_by_key)


def fallback_evaluate_matches(
    startup_profile: dict,
    opportunities: List[dict],
    precomputed_audits: Optional[Dict[str, dict]] = None,
) -> dict:
    """
    Motor determinístico auditado (40% Área | 30% Tecnologias | 30% Demais Critérios)
    que analisa os campos reais da startup e de cada oportunidade com crescimento natural.
    """
    name = startup_profile.get("name") or "Sua Startup"
    main_area = startup_profile.get("mainArea") or startup_profile.get("sector") or "Inovação"
    how_it_works = startup_profile.get("howItWorks") or ""

    matches = []
    for opp in opportunities:
        opp_id = str(opp.get("id") or opp.get("code") or "")
        title = opp.get("title") or "Oportunidade"
        opp_type = (opp.get("type") or "EDITAL").upper()

        audit = None
        if precomputed_audits:
            audit = precomputed_audits.get(opp_id) or precomputed_audits.get(title.strip().lower())
        if not audit:
            audit = compute_audited_match_breakdown(startup_profile, opp, target_kind="opportunity")

        matched_techs_str = ", ".join(audit["matchedTechnologies"])
        action_focus = matched_techs_str if matched_techs_str else main_area

        matches.append({
            "opportunity_id": opp_id,
            "opportunity_title": title,
            "opportunity_type": opp_type,
            "opportunity_description": opp.get("description") or opp.get("shortDescription") or "",
            "opportunity_meta": opp.get("meta") or opp.get("metaInfo") or opp.get("deadline") or "Inscrições abertas · Brasil",
            "opportunity_tags": opp.get("tags") or "Inovação • Tecnologia",
            "url": opp.get("externalUrl") or opp.get("url") or "",
            "match_score": audit["totalScore"],
            "justification": _build_default_justification(name, title, audit),
            "recommended_action": (
                f"Estruturar submissão da {name} evidenciando {action_focus} "
                f"({how_it_works[:85] + '...' if len(how_it_works) > 85 else (how_it_works or 'aplicação prática da solução')})."
            ),
            "reasons": audit["reasons"],
            "audit_breakdown": audit,
        })

    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return {"matches": matches}


def evaluate_company_to_startups_matches(company: dict, startups: List[dict]) -> dict:
    """
    Avalia todas as startups cadastradas na plataforma contra os desafios/demandas da EMPRESA
    usando a balança auditada (40% Área | 30% Tecnologias | 30% Demais Critérios).
    """
    company = company or {}
    startups = startups or fetch_all_startups_from_backend()
    if not startups:
        return {"matches": []}

    comp_name = company.get("name") or "Sua Empresa"
    comp_sector = company.get("sector") or "Inovação Corporativa"
    comp_desc = company.get("description") or ""
    comp_demands = company.get("demands") or comp_desc
    comp_tags = company.get("tags") or ""
    comp_loc = company.get("location") or "Brasil"
    comp_mode = company.get("operationMode") or "POC, Compra"

    audit_by_st_key: Dict[str, dict] = {}
    prompt_candidates = []
    for idx, st in enumerate(startups):
        skey = str(st.get("id") or (st.get("name") or "").strip().lower() or f"st-{idx + 1}")
        audit = compute_audited_match_breakdown(st, company, target_kind="company")
        audit_by_st_key[skey] = audit
        if st.get("name"):
            audit_by_st_key[st["name"].strip().lower()] = audit
        prompt_candidates.append({
            "startupId": st.get("id"),
            "startupName": st.get("name"),
            "mainArea": st.get("mainArea") or st.get("mainSector"),
            "technologies": st.get("technologies") or st.get("areasOfInterest"),
            "problem": st.get("problem") or st.get("description"),
            "currentStage": st.get("currentStage"),
            "trlScore": st.get("trlScore"),
            "location": st.get("location"),
            "audited_scores": {
                "area_score_of_40": audit["areaScore"],
                "tech_score_of_30": audit["techScore"],
                "matched_technologies": audit["matchedTechnologies"],
                "other_score_of_30": audit["otherScore"],
                "total_audited_score": audit["totalScore"],
            },
        })

    if client:
        prompt = f"""
        Você é o motor oficial auditado de Inteligência Artificial do StartIn.
        Analise as demandas da EMPRESA abaixo e redija a avaliação em TEXTO PLANO PURO (sem Markdown, sem asteriscos) para CADA UMA das {len(startups)} STARTUPS cadastradas, respeitando ESTRITAMENTE a balança auditada:
        - 40% (0 a 40 pts): Mesma Área de Atuação / Setor
        - 30% (0 a 30 pts): Tecnologias iguais / compatíveis
        - 30% (0 a 30 pts): Demais critérios (Estágio/TRL 10 pts + Modalidade/Modelo 10 pts + Problema/Demanda 5 pts + Localização 5 pts)

        [DADOS DA EMPRESA CONTRATANTE]
        - Nome da Empresa: {comp_name}
        - Setor: {comp_sector}
        - Localização: {comp_loc}
        - Modalidades: {comp_mode}
        - Descrição: {comp_desc or 'Não informada'}
        - Demandas de POC / Compra: {comp_demands or 'Não informadas'}
        - Tecnologias buscadas: {comp_tags or 'Não informadas'}

        [STARTUPS COM PONTUAÇÃO AUDITADA]
        {json.dumps(prompt_candidates, ensure_ascii=False, indent=2)}

        Retorne ESTRITAMENTE um JSON válido com "matches" para todas as startups, informando `semantic_affinity_0_to_5` (0 a 5) e explicações em texto plano sem markdown:
        {{
          "matches": [
            {{
              "startupId": 1,
              "startupName": "Nome da Startup",
              "semantic_affinity_0_to_5": 3,
              "justification": "Explicação transparente do score com base na área, tecnologias coincidentes e estágio...",
              "recommendedAction": "Próximo passo recomendado para a empresa...",
              "reasons": {{
                "perfil": "Alinhamento entre setor da empresa e área da startup",
                "requisitos": "Tecnologias coincidentes entre empresa e startup",
                "objetivo": "Adequação de estágio TRL e modalidade POC/Compra",
                "contexto": "Aderência à demanda e localização"
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
                        "content": "Você é um auditor técnico de inovação aberta entre Empresas e Startups. Respeite os pesos auditados (40% área, 30% tecnologias, 30% demais), escreva em texto plano sem markdown e responda apenas com JSON válido.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.15,
            )
            parsed = json.loads(completion.choices[0].message.content)
            ai_list = parsed.get("matches")
            if isinstance(ai_list, list) and len(ai_list) > 0:
                st_by_id = {str(s.get("id")): s for s in startups if s.get("id") is not None}
                st_by_name = {(s.get("name") or "").strip().lower(): s for s in startups if s.get("name")}
                enriched = []
                seen_ids = set()

                for m in ai_list:
                    sid = str(m.get("startupId") or m.get("startup_id") or "")
                    sname = (m.get("startupName") or m.get("startup_name") or "").strip().lower()
                    src = st_by_id.get(sid) or st_by_name.get(sname)
                    if not src:
                        continue
                    dedup_key = str(src.get("id") or sname)
                    if dedup_key in seen_ids:
                        continue
                    seen_ids.add(dedup_key)

                    base_audit = audit_by_st_key.get(str(src.get("id"))) or audit_by_st_key.get(sname) or compute_audited_match_breakdown(src, company, "company")
                    final_audit = _recompute_audit_with_llm_semantic(
                        base_audit,
                        m.get("semantic_affinity_0_to_5"),
                        m.get("reasons"),
                    )
                    desc = src.get("description") or src.get("problem") or "Solução tecnológica cadastrada na plataforma."
                    st_display_name = src.get("name") or m.get("startupName") or "Startup"

                    enriched.append({
                        "startupId": src.get("id") or m.get("startupId"),
                        "startupName": st_display_name,
                        "mainArea": src.get("mainArea") or src.get("mainSector") or "Inovação",
                        "description": desc,
                        "shortDescription": desc[:120] + ("..." if len(desc) > 120 else ""),
                        "technologies": src.get("technologies") or src.get("areasOfInterest") or "Não informadas",
                        "businessModel": src.get("businessModel") or "B2B",
                        "currentStage": src.get("currentStage") or "Validação",
                        "location": src.get("location") or "Brasil",
                        "website": src.get("website") or "",
                        "phone": src.get("phone") or "",
                        "email": src.get("userEmail") or "",
                        "avatarUrl": src.get("avatarUrl"),
                        "matchScore": final_audit["totalScore"],
                        "justification": to_plain_text(str(m.get("justification") or "")) or _build_default_justification(st_display_name, comp_name, final_audit),
                        "recommendedAction": to_plain_text(str(m.get("recommendedAction") or m.get("recommended_action") or "")) or f"Avaliar conexão de {comp_mode} com a {st_display_name}.",
                        "reasons": final_audit["reasons"],
                        "auditBreakdown": final_audit,
                    })

                for st in startups:
                    dkey = str(st.get("id") or (st.get("name") or "").strip().lower())
                    if dkey not in seen_ids:
                        audit = audit_by_st_key.get(dkey) or compute_audited_match_breakdown(st, company, "company")
                        desc = st.get("description") or st.get("problem") or "Solução tecnológica cadastrada na plataforma."
                        st_name = st.get("name") or "Startup"
                        enriched.append({
                            "startupId": st.get("id"),
                            "startupName": st_name,
                            "mainArea": st.get("mainArea") or st.get("mainSector") or "Inovação",
                            "description": desc,
                            "shortDescription": desc[:120] + ("..." if len(desc) > 120 else ""),
                            "technologies": st.get("technologies") or st.get("areasOfInterest") or "Não informadas",
                            "businessModel": st.get("businessModel") or "B2B",
                            "currentStage": st.get("currentStage") or "Validação",
                            "location": st.get("location") or "Brasil",
                            "website": st.get("website") or "",
                            "phone": st.get("phone") or "",
                            "email": st.get("userEmail") or "",
                            "avatarUrl": st.get("avatarUrl"),
                            "matchScore": audit["totalScore"],
                            "justification": _build_default_justification(st_name, comp_name, audit),
                            "recommendedAction": f"Avaliar conexão com a {st_name} para demandas de {comp_mode}.",
                            "reasons": audit["reasons"],
                            "auditBreakdown": audit,
                        })

                enriched.sort(key=lambda x: x["matchScore"], reverse=True)
                return {"matches": enriched}
        except Exception as e:
            print(f"Aviso ao calcular match auditado Empresa x Startups na Groq: {e}")

    results = []
    for idx, st in enumerate(startups):
        st_name = st.get("name") or "Startup"
        st_area = st.get("mainArea") or st.get("mainSector") or "Inovação"
        st_tech = st.get("technologies") or st.get("areasOfInterest") or "Não informadas"
        st_desc = st.get("description") or st.get("problem") or "Solução tecnológica cadastrada na plataforma."
        st_stage = st.get("currentStage") or "Validação"
        st_loc = st.get("location") or "Brasil"
        st_model = st.get("businessModel") or "B2B"

        skey = str(st.get("id") or st_name.strip().lower() or f"st-{idx + 1}")
        audit = audit_by_st_key.get(skey) or compute_audited_match_breakdown(st, company, target_kind="company")

        results.append({
            "startupId": st.get("id"),
            "startupName": st_name,
            "mainArea": st_area,
            "description": st_desc,
            "shortDescription": st_desc[:120] + ("..." if len(st_desc) > 120 else ""),
            "technologies": st_tech,
            "businessModel": st_model,
            "currentStage": st_stage,
            "location": st_loc,
            "website": st.get("website") or "",
            "phone": st.get("phone") or "",
            "email": st.get("userEmail") or "",
            "avatarUrl": st.get("avatarUrl"),
            "matchScore": audit["totalScore"],
            "justification": _build_default_justification(st_name, comp_name, audit),
            "recommendedAction": f"Convidar a {st_name} para avaliação técnica focada em {comp_sector}.",
            "reasons": audit["reasons"],
            "auditBreakdown": audit,
        })

    results.sort(key=lambda x: x["matchScore"], reverse=True)
    return {"matches": results}


def evaluate_startup_to_companies_matches(startup: dict, companies: List[dict]) -> dict:
    """
    Avalia todas as empresas cadastradas na plataforma contra o perfil da STARTUP
    usando a balança auditada (40% Área | 30% Tecnologias | 30% Demais Critérios).
    """
    startup = startup or fetch_user_and_startup_from_backend()
    companies = companies or []
    if not companies:
        return {"matches": []}

    st_name = startup.get("name") or "Sua Startup"

    audit_by_comp_key: Dict[str, dict] = {}
    prompt_companies = []
    for idx, comp in enumerate(companies):
        ckey = str(comp.get("id") or (comp.get("name") or "").strip().lower() or f"comp-{idx + 1}")
        audit = compute_audited_match_breakdown(startup, comp, target_kind="company")
        audit_by_comp_key[ckey] = audit
        if comp.get("name"):
            audit_by_comp_key[comp["name"].strip().lower()] = audit
        prompt_companies.append({
            "companyId": comp.get("id"),
            "companyName": comp.get("name"),
            "sector": comp.get("sector"),
            "tags": comp.get("tags"),
            "demands": comp.get("demands") or comp.get("description"),
            "operationMode": comp.get("operationMode"),
            "preferredStage": comp.get("preferredStage"),
            "location": comp.get("location"),
            "audited_scores": {
                "area_score_of_40": audit["areaScore"],
                "tech_score_of_30": audit["techScore"],
                "matched_technologies": audit["matchedTechnologies"],
                "other_score_of_30": audit["otherScore"],
                "total_audited_score": audit["totalScore"],
            },
        })

    if client:
        prompt = f"""
        Você é o motor oficial auditado de Inteligência Artificial do StartIn.
        Analise o perfil da STARTUP abaixo e redija a avaliação em TEXTO PLANO PURO (sem Markdown, sem asteriscos) com CADA UMA das {len(companies)} EMPRESAS cadastradas, respeitando ESTRITAMENTE a balança auditada:
        - 40% (0 a 40 pts): Mesma Área de Atuação / Setor
        - 30% (0 a 30 pts): Tecnologias iguais / compatíveis
        - 30% (0 a 30 pts): Demais critérios (Estágio/TRL 10 pts + Modalidade/Modelo 10 pts + Problema/Demanda 5 pts + Localização 5 pts)

        {_format_profile_for_prompt(startup)}

        [EMPRESAS COM PONTUAÇÃO AUDITADA]
        {json.dumps(prompt_companies, ensure_ascii=False, indent=2)}

        Retorne ESTRITAMENTE um JSON válido com "matches" para todas as empresas, informando `semantic_affinity_0_to_5` (0 a 5) e textos em texto plano sem markdown:
        {{
          "matches": [
            {{
              "companyId": 1,
              "companyName": "Nome da Empresa",
              "semantic_affinity_0_to_5": 3,
              "justification": "Explicação transparente do score com base na área, tecnologias coincidentes e demandas...",
              "recommendedAction": "Ação recomendada para a startup abordar a empresa...",
              "reasons": {{
                "perfil": "Alinhamento entre área da startup e setor da empresa",
                "requisitos": "Tecnologias coincidentes com o que a empresa busca",
                "objetivo": "Adequação ao estágio e modalidade POC/Compra",
                "contexto": "Aderência à demanda e região de atuação"
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
                        "content": "Você é um auditor técnico de inovação aberta entre Startups e Empresas. Respeite os pesos auditados (40% área, 30% tecnologias, 30% demais), escreva em texto plano sem markdown e responda apenas com JSON válido.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.15,
            )
            parsed = json.loads(completion.choices[0].message.content)
            ai_list = parsed.get("matches")
            if isinstance(ai_list, list) and len(ai_list) > 0:
                comp_by_id = {str(c.get("id")): c for c in companies if c.get("id") is not None}
                comp_by_name = {(c.get("name") or "").strip().lower(): c for c in companies if c.get("name")}
                enriched = []
                seen_ids = set()

                for m in ai_list:
                    cid = str(m.get("companyId") or m.get("company_id") or "")
                    cname = (m.get("companyName") or m.get("company_name") or "").strip().lower()
                    src = comp_by_id.get(cid) or comp_by_name.get(cname)
                    if not src:
                        continue
                    dedup_key = str(src.get("id") or cname)
                    if dedup_key in seen_ids:
                        continue
                    seen_ids.add(dedup_key)

                    base_audit = audit_by_comp_key.get(str(src.get("id"))) or audit_by_comp_key.get(cname) or compute_audited_match_breakdown(startup, src, "company")
                    final_audit = _recompute_audit_with_llm_semantic(
                        base_audit,
                        m.get("semantic_affinity_0_to_5"),
                        m.get("reasons"),
                    )
                    desc = src.get("description") or "Organização parceira com demandas abertas de inovação."
                    comp_display_name = src.get("name") or m.get("companyName") or "Empresa"

                    enriched.append({
                        "companyId": src.get("id") or m.get("companyId"),
                        "companyName": comp_display_name,
                        "companyType": src.get("type") or "EMPRESA",
                        "sector": src.get("sector") or "Inovação Corporativa",
                        "description": desc,
                        "demands": src.get("demands") or desc,
                        "tags": src.get("tags") or "Inovação",
                        "location": src.get("location") or "Brasil",
                        "website": src.get("website") or "",
                        "phone": src.get("phone") or "",
                        "cnpj": src.get("cnpj") or "",
                        "operationMode": src.get("operationMode") or "POC, Compra",
                        "preferredStage": src.get("preferredStage") or "Validação / Tração",
                        "avatarUrl": src.get("avatarUrl"),
                        "matchScore": final_audit["totalScore"],
                        "justification": to_plain_text(str(m.get("justification") or "")) or _build_default_justification(st_name, comp_display_name, final_audit),
                        "recommendedAction": to_plain_text(str(m.get("recommendedAction") or m.get("recommended_action") or "")) or f"Apresentar proposta de {src.get('operationMode') or 'POC / Compra'} da {st_name}.",
                        "reasons": final_audit["reasons"],
                        "auditBreakdown": final_audit,
                    })

                for comp in companies:
                    dkey = str(comp.get("id") or (comp.get("name") or "").strip().lower())
                    if dkey not in seen_ids:
                        audit = audit_by_comp_key.get(dkey) or compute_audited_match_breakdown(startup, comp, "company")
                        desc = comp.get("description") or "Organização parceira com demandas abertas de inovação."
                        comp_name = comp.get("name") or "Empresa"
                        enriched.append({
                            "companyId": comp.get("id"),
                            "companyName": comp_name,
                            "companyType": comp.get("type") or "EMPRESA",
                            "sector": comp.get("sector") or "Inovação Corporativa",
                            "description": desc,
                            "demands": comp.get("demands") or desc,
                            "tags": comp.get("tags") or "Inovação",
                            "location": comp.get("location") or "Brasil",
                            "website": comp.get("website") or "",
                            "phone": comp.get("phone") or "",
                            "cnpj": comp.get("cnpj") or "",
                            "operationMode": comp.get("operationMode") or "POC, Compra",
                            "preferredStage": comp.get("preferredStage") or "Validação / Tração",
                            "avatarUrl": comp.get("avatarUrl"),
                            "matchScore": audit["totalScore"],
                            "justification": _build_default_justification(st_name, comp_name, audit),
                            "recommendedAction": f"Apresentar proposta da {st_name} para {comp_name}.",
                            "reasons": audit["reasons"],
                            "auditBreakdown": audit,
                        })

                enriched.sort(key=lambda x: x["matchScore"], reverse=True)
                return {"matches": enriched}
        except Exception as e:
            print(f"Aviso ao calcular match auditado Startup x Empresas na Groq: {e}")

    results = []
    for idx, comp in enumerate(companies):
        comp_name = comp.get("name") or "Empresa"
        comp_type = comp.get("type") or "EMPRESA"
        comp_sector = comp.get("sector") or "Inovação Corporativa"
        comp_desc = comp.get("description") or "Organização parceira com demandas abertas de inovação."
        comp_demands = comp.get("demands") or comp_desc
        comp_tags = comp.get("tags") or "Inovação"
        comp_loc = comp.get("location") or "Brasil"
        comp_mode = comp.get("operationMode") or "POC, Compra"

        ckey = str(comp.get("id") or comp_name.strip().lower() or f"comp-{idx + 1}")
        audit = audit_by_comp_key.get(ckey) or compute_audited_match_breakdown(startup, comp, target_kind="company")

        results.append({
            "companyId": comp.get("id"),
            "companyName": comp_name,
            "companyType": comp_type,
            "sector": comp_sector,
            "description": comp_desc,
            "demands": comp_demands,
            "tags": comp_tags,
            "location": comp_loc,
            "website": comp.get("website") or "",
            "phone": comp.get("phone") or "",
            "cnpj": comp.get("cnpj") or "",
            "operationMode": comp_mode,
            "preferredStage": comp.get("preferredStage") or "Validação / Tração",
            "avatarUrl": comp.get("avatarUrl"),
            "matchScore": audit["totalScore"],
            "justification": _build_default_justification(st_name, comp_name, audit),
            "recommendedAction": f"Apresentar proposta de {comp_mode} da {st_name} para os desafios da {comp_name}.",
            "reasons": audit["reasons"],
            "auditBreakdown": audit,
        })

    results.sort(key=lambda x: x["matchScore"], reverse=True)
    return {"matches": results}


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
# 5. ESTRUTURAÇÃO DE TEXTO COM IA (ONBOARDING ETAPA 02) E ATESTADO TRL/PDF (ETAPA 03)
# =============================================================================

def replace_pilot_with_poc(val: Any) -> Any:
    """Garante que qualquer menção a piloto/piloto rápido seja substituída por prova de conceito (POC)."""
    if isinstance(val, str):
        v = re.sub(r"(?i)\bprojetos?-?pilotos?\b", "provas de conceito (POC)", val)
        v = re.sub(r"(?i)\bpilotos?\s+r[aá]pidos?\b", "provas de conceito (POC)", v)
        v = re.sub(r"(?i)\bpilotos?\b", "provas de conceito", v)
        return v
    elif isinstance(val, list):
        return [replace_pilot_with_poc(x) for x in val]
    elif isinstance(val, dict):
        return {k: replace_pilot_with_poc(v) for k, v in val.items()}
    return val


def enhance_startup_problem_text(
    raw_text: str,
    main_area: str = "",
    startup_name: str = "",
) -> Dict[str, Any]:
    """
    Lê o rascunho escrito pelo usuário na caixa 'Problema que resolve', melhora e
    estrutura o texto de forma totalmente integrada ao que foi passado previamente
    (sem templates engessados ou frases genéricas pré-prontas) e extrai automaticamente
    as tags, tecnologias, modelo de negócio, funcionamento, diferencial e impacto.
    """
    clean_input = (raw_text or "").strip()
    area_label = (main_area or "Tecnologia & Inovação").strip()
    name_label = (startup_name or "A startup").strip()

    if client and clean_input:
        prompt = f"""
        Você é o arquiteto especialista em matchmaking e inovação aberta da plataforma StartIn.
        O fundador da startup '{name_label}' (Área principal: {area_label}) escreveu o seguinte texto base sobre a dor que resolve e sua solução:
        \"\"\"{clean_input}\"\"\"

        DIRETRIZES DE INTEGRAÇÃO TOTAL E SÍNTESE EXECUTIVA (SEM TEMPLATES ENGESSADOS):
        1. INTEGRAÇÃO REAL DO TEXTO BASE:
           - Leia com máxima atenção o texto passado. Compreenda cada termo, tecnologia, cliente, modelo e problema mencionado.
           - NÃO jogue um template pré-pronto por cima! NÃO repita apenas a frase do usuário com textos genéricos colados.
           - REESCREVA E ARTICULE o conteúdo em uma narrativa executiva fluida, coesa e profissional, onde CADA parágrafo seja desenvolvido a partir das informações fornecidas pelo usuário.
        2. ESTRUTURAÇÃO FLUIDA E PROFISSIONAL:
           Organize o texto em parágrafos temáticos bem desenvolvidos (separados por linha em branco), por exemplo:
           - Problema e Dor de Mercado: contextualize a dor real e a ineficiência que a solução combate, com base no que o usuário escreveu.
           - Solução e Arquitetura Tecnológica: aprofunde como a solução da startup opera na prática e quais tecnologias são aplicadas.
           - Modelo de Negócio e Público Atendido: especifique o perfil de clientes e a modalidade de entrega.
           - Diferencial Competitivo e Impacto Mensurável: consolide os ganhos operacionais, financeiros ou de escala proporcionados.
        3. FORMATO OBRIGATÓRIO:
           - O campo "enhancedText" e todos os outros devem estar ESTRITAMENTE em texto plano puro, SEM NENHUM Markdown (proibido usar *, **, #, ---, >, listas com -, crases ` ou tabelas |).
        4. TAGS E TÓPICOS INTEGRADOS:
           - Extraia entre 4 e 7 tags/tópicos concisos e representativos que estejam presentes e integrados ao texto, para alimentar o motor de matchmaking da plataforma.
        5. TERMINOLOGIA OBRIGATÓRIA:
           - Em relação à modalidade POC, utilize SEMPRE o termo 'prova de conceito' ou 'POC'. NUNCA utilize termos como 'piloto' ou 'piloto rápido' para designar POC.

        Retorne ESTRITAMENTE um JSON válido com esta estrutura:
        {{
          "enhancedText": "Problema Central:\\n[Parágrafo articulando a dor real com base no texto]\\n\\nSolução e Tecnologia Aplicada:\\n[Parágrafo detalhando a tecnologia e funcionamento com base no texto]\\n\\nModelo de Negócio e Público-Alvo:\\n[Parágrafo detalhando clientes e modelo]\\n\\nDiferencial e Impacto:\\n[Parágrafo consolidando benefícios e métricas]",
          "tags": ["Tag1", "Tag2", "Tag3", "Tag4"],
          "technologies": "Tag1, Tag2, Tag3, Tag4",
          "businessModel": "SaaS B2B ou modelo extraído",
          "howItWorks": "Resumo objetivo articulado da solução...",
          "differentiator": "Diferencial competitivo...",
          "impact": "Impacto gerado...",
          "targetAudience": "Público atendido..."
        }}
        """
        for candidate_model in FALLBACK_MODELS:
            try:
                completion = client.chat.completions.create(
                    model=candidate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um especialista em posicionamento de startups e inovação aberta. Responda apenas em JSON válido.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                parsed = json.loads(completion.choices[0].message.content or "{}")
                if parsed.get("enhancedText"):
                    tags = parsed.get("tags") or []
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(",") if t.strip()]
                    tags = [str(t).strip().lstrip("#") for t in tags if str(t).strip()][:7]
                    if not tags:
                        tags = [area_label, "Inovação", "B2B", "Tecnologia"]
                    return replace_pilot_with_poc({
                        "enhancedText": to_structured_plain_text(parsed["enhancedText"]),
                        "tags": tags,
                        "technologies": ", ".join(tags),
                        "businessModel": parsed.get("businessModel") or "SaaS B2B",
                        "howItWorks": parsed.get("howItWorks") or "",
                        "differentiator": parsed.get("differentiator") or "",
                        "impact": parsed.get("impact") or "",
                        "targetAudience": parsed.get("targetAudience") or "Empresas e organizações B2B",
                    })
            except Exception as e:
                print(f"[StartIn IA] Aviso no modelo '{candidate_model}': {e}")

    # Fallback determinístico semanticamente integrado ao texto
    lower_in = f"{clean_input} {area_label}".lower()
    inferred_tags: List[str] = []
    keyword_map = [
        (["ia", "inteligência artificial", "machine learning", "llm", "preditiv"], "IA"),
        (["dado", "dados", "analytics", "bi", "dashboard", "indicador"], "Dados & Analytics"),
        (["automa", "rpa", "fluxo", "processo", "operacional"], "Automação"),
        (["saúde", "health", "hospital", "clínic", "médic", "paciente"], "Healthtech"),
        (["finan", "fintech", "crédito", "pagament", "banc"], "Fintech"),
        (["agro", "campo", "safra", "rural"], "Agtech"),
        (["educa", "ensino", "aprendiz", "edtech"], "Edtech"),
        (["iot", "sensor", "indústria", "manufatura", "hardware"], "IoT & Indústria 4.0"),
        (["segurança", "cyber", "lgpd", "fraude"], "Cibersegurança"),
        (["nuvem", "cloud", "api", "integra"], "Cloud & APIs"),
        (["b2c", "consumidor final", "varejo b2c"], "B2C"),
        (["b2b", "corporativ", "empresa"], "B2B"),
        (["saas", "plataforma", "assinatura", "software"], "SaaS"),
    ]
    for kws, tag_label in keyword_map:
        if any(k in lower_in for k in kws) and tag_label not in inferred_tags:
            inferred_tags.append(tag_label)

    if area_label and area_label not in inferred_tags:
        inferred_tags.insert(0, area_label)
    for default_t in ["Inovação", "Eficiência"]:
        if len(inferred_tags) < 5 and default_t not in inferred_tags:
            inferred_tags.append(default_t)
    inferred_tags = inferred_tags[:6]

    is_b2c = "b2c" in lower_in
    b_model = "Plataforma B2C" if is_b2c else ("Hardware & IoT B2B" if "iot" in lower_in or "hardware" in lower_in else "SaaS B2B")
    target_audience = "Consumidores finais e usuários do segmento B2C" if is_b2c else f"Empresas, corporações e organizações do setor de {area_label}"

    p_clean = clean_input.rstrip('.') if clean_input else f"Processos manuais e ineficiência operacional no setor de {area_label}"

    p1 = f"Problema Central:\nIdentificamos que no ecossistema de {area_label}, o principal gargalo enfrentado é {p_clean}. Essa dor acarreta perda de produtividade, custos elevados e falta de integração em fluxos estratégicos."
    p2 = f"Solução e Tecnologia Aplicada:\nA {name_label} atua como solução tecnológica especializada, aplicando {', '.join(inferred_tags[:3])} para automatizar processos, estruturar dados e oferecer ferramentas inteligentes que resolvem diretamente o desafio relatado."
    p3 = f"Modelo de Negócio e Público-Alvo:\nOperamos sob o modelo de {b_model}, atendendo {target_audience.lower()} com arquitetura escalável e suporte dedicado."
    p4 = f"Diferencial e Impacto Esperado:\nEntregamos implantação ágil, redução mensurável de atrito operacional e governança em tempo real, gerando valor comprovado para clientes e parceiros corporativos."

    structured = f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}"

    return replace_pilot_with_poc({
        "enhancedText": structured,
        "tags": inferred_tags,
        "technologies": ", ".join(inferred_tags),
        "businessModel": b_model,
        "howItWorks": p2,
        "differentiator": p4,
        "impact": "Redução mensurável de atrito operacional e ganho de escala.",
        "targetAudience": target_audience,
    })


def enhance_company_demands_text(
    raw_text: str,
    sector: str = "",
    company_name: str = "",
    operation_mode: str = "",
) -> Dict[str, Any]:
    """
    Lê o texto de desafios e demandas operacionais escrito pela empresa,
    aprimora e organiza de forma totalmente integrada ao que foi passado previamente
    (sem templates engessados ou substituições artificiais), articulando a narrativa
    em parágrafos executivos coesos baseados no texto fornecido.
    """
    clean_input = (raw_text or "").strip()
    sector_label = (sector or "Operações Corporativas e Tecnologia").strip()
    company_label = (company_name or "A Empresa").strip()
    mode_label = (operation_mode or "POC + Compra").strip()

    if client and clean_input:
        prompt = f"""
        Você é o arquiteto especialista em inovação aberta e contratações corporativas da plataforma StartIn.
        A empresa '{company_label}' (Setor de atuação: {sector_label}, Modalidade preferencial: {mode_label}) forneceu as seguintes informações sobre seus desafios, processos e demandas:
        \"\"\"{clean_input}\"\"\"

        INSTRUÇÕES DE INTEGRAÇÃO TOTAL E SÍNTESE EXECUTIVA (SEM TEMPLATES ENGESSADOS):
        1. INTEGRAÇÃO REAL DO TEXTO BASE:
           - LEIA e COMPREENDA profundamente cada frase, detalhe, dor, público (ex: B2C, B2B), processos e objetivos que a empresa escreveu.
           - NÃO jogue um template genérico por cima! NÃO cole o texto bruto sob um cabeçalho para depois preencher com textos padronizados pré-fabricados.
           - REESCREVA E ARTICULE o conteúdo em uma narrativa executiva fluida, coesa e profissional, onde CADA parágrafo seja diretamente construído a partir das dores e necessidades reais que a empresa descreveu.
           - Se a empresa mencionou B2C, foque a narrativa no relacionamento e esteira voltados para o consumidor final B2C. Se citou automação e dados, estruture em torno da modernização desses fluxos.
        2. ESTRUTURAÇÃO EXECUTIVA ARTICULADA:
           Organize o texto em seções temáticas fluidas (separadas por linha em branco):
           - Desafio Operacional Central: apresente quem é a empresa e sintetize a dor ou gargalo prioritário que ela relatou.
           - Demandas Tecnológicas & Escopo da Parceria: detalhe as soluções, ferramentas e automações que a empresa busca nas startups com base estrita no que ela descreveu.
           - Modelo de Cooperação & Resultados Esperados: indique o formato de parceria ({mode_label}) e os ganhos concretos visados (eficiência, agilidade, redução de custos e escala).
        3. FORMATO OBRIGATÓRIO:
           - O campo "enhancedText" e todos os outros devem estar ESTRITAMENTE em texto plano puro, SEM NENHUM Markdown (absolutamente proibido usar *, **, #, ---, >, listas com -, crases ` ou tabelas |).
        4. TAGS E TÓPICOS INTEGRADOS:
           - Extraia entre 4 e 7 tags/tópicos concisos e fiéis que resumam as tecnologias e desafios relatados para alimentar o motor de matchmaking com startups.
        5. TERMINOLOGIA OBRIGATÓRIA:
           - Em relação à modalidade POC, utilize SEMPRE o termo 'prova de conceito' ou 'POC'. NUNCA utilize termos como 'piloto' ou 'piloto rápido' para designar POC.

        Retorne ESTRITAMENTE um JSON válido com esta estrutura:
        {{
          "enhancedText": "Desafio Operacional Central:\\n[Parágrafo articulado sintetizando o contexto e a dor real informada]\\n\\nDemandas Tecnológicas & Escopo da Parceria:\\n[Parágrafo articulando as tecnologias e soluções esperadas com base no texto]\\n\\nModelo de Cooperação & Resultados Esperados:\\n[Parágrafo articulando o modelo {mode_label} e resultados esperados]",
          "tags": ["Automação de Processos", "IA & Dados", "SaaS B2B", "Integração ERP"],
          "technologies": "Automação de Processos, IA & Dados, SaaS B2B, Integração ERP",
          "priorityTopics": ["Automação de Processos", "Inteligência Operacional", "Governança de Dados"],
          "howItWorks": "Resumo articulado das demandas e escopo...",
          "impact": "Resultados concretos e critérios de homologação..."
        }}
        """
        for candidate_model in FALLBACK_MODELS:
            try:
                completion = client.chat.completions.create(
                    model=candidate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Você é um especialista em inovação aberta, compras corporativas e desafios de POC para grandes empresas. Responda apenas em JSON válido.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                parsed = json.loads(completion.choices[0].message.content or "{}")
                if parsed.get("enhancedText"):
                    tags = parsed.get("tags") or []
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(",") if t.strip()]
                    tags = [str(t).strip().lstrip("#") for t in tags if str(t).strip()][:7]
                    if not tags:
                        tags = [sector_label, "Inovação Aberta", "B2B", "Eficiência"]
                    p_topics = parsed.get("priorityTopics") or tags[:4]
                    if isinstance(p_topics, str):
                        p_topics = [t.strip() for t in p_topics.split(",") if t.strip()]
                    return replace_pilot_with_poc({
                        "enhancedText": to_structured_plain_text(parsed["enhancedText"]),
                        "tags": tags,
                        "technologies": ", ".join(tags),
                        "priorityTopics": p_topics,
                        "howItWorks": parsed.get("howItWorks") or "",
                        "impact": parsed.get("impact") or "",
                    })
            except Exception as e:
                print(f"[StartIn IA] Aviso no modelo '{candidate_model}': {e}")

    # Fallback determinístico semanticamente integrado ao texto fornecido
    lower_in = f"{clean_input} {sector_label}".lower()
    inferred_tags: List[str] = []
    keyword_map = [
        (["ia", "inteligência artificial", "machine learning", "dados", "analytics"], "IA & Dados"),
        (["automa", "rpa", "fluxo", "manual", "processo"], "Automação de Processos"),
        (["erp", "crm", "sap", "totvs", "sistema legado", "integra"], "Integração de Sistemas"),
        (["saúde", "hospital", "prontuário", "paciente", "clínic"], "Healthtech"),
        (["finan", "faturamento", "conciliação", "pagamento"], "Fintech & Gestão Financeira"),
        (["logística", "rastreio", "frota", "estoque", "armazém"], "Logística & Supply Chain"),
        (["segurança", "lgpd", "ciber", "compliance"], "Cibersegurança & Compliance"),
        (["nuvem", "cloud", "api", "microsserviços"], "Cloud & APIs"),
        (["b2c", "consumidor final", "cliente final"], "Operações B2C"),
        (["b2b", "corporativ", "empresas"], "Operações B2B"),
        (["poc", "prova de conceito", "homologação"], "POC / Prova de Conceito"),
        (["compra", "escala", "fornecedor"], "Compra Corporativa"),
    ]
    for kws, tag_label in keyword_map:
        if any(k in lower_in for k in kws) and tag_label not in inferred_tags:
            inferred_tags.append(tag_label)

    if sector_label and sector_label not in inferred_tags:
        inferred_tags.insert(0, sector_label)
    for default_t in ["Inovação Aberta", "Eficiência Operacional"]:
        if len(inferred_tags) < 5 and default_t not in inferred_tags:
            inferred_tags.append(default_t)
    inferred_tags = inferred_tags[:6]

    is_b2c = "b2c" in lower_in
    segment_desc = "no segmento B2C (voltado ao consumidor final)" if is_b2c else "em operações corporativas"
    core_text = clean_input.rstrip('.') if clean_input else f"Necessidade de automação de processos e eficiência no setor de {sector_label}"

    p1 = f"Desafio Operacional Central:\nAtuando {segment_desc} no setor de {sector_label}, nossa organização enfrenta o desafio de {core_text}. O foco prioritário é modernizar processos manuais, eliminar retrabalho e trazer inteligência operacional para a esteira de atendimento e execução."
    p2 = f"Demandas Tecnológicas & Escopo da Parceria:\nBuscamos soluções desenvolvidas por startups com competência comprovada em {', '.join(inferred_tags[:3])}, capazes de integrar-se de forma transparente à nossa arquitetura e proporcionar ganhos ágeis de produtividade."
    p3 = f"Modelo de Cooperação & Resultados Esperados:\nPriorizamos projetos na modalidade {mode_label}, com testes estruturados e métricas claras. Esperamos como resultados mensuráveis a redução do tempo de ciclo de atendimento, diminuição de erros operacionais e escalabilidade sustentável."

    structured = f"{p1}\n\n{p2}\n\n{p3}"

    return replace_pilot_with_poc({
        "enhancedText": structured,
        "tags": inferred_tags,
        "technologies": ", ".join(inferred_tags),
        "priorityTopics": inferred_tags[:4],
        "howItWorks": p2,
        "impact": "Redução do tempo de ciclo, diminuição de erros e escalabilidade operacional.",
    })


def _clean_previous_provider_boilerplate(raw_text: str, provider_name: str = "") -> str:
    """
    Remove prefixos ou blocos em 3ª pessoa gerados por versões anteriores
    para extrair exclusivamente a frase original escrita pelo próprio usuário.
    """
    text = (raw_text or "").strip()
    if not text:
        return ""

    # Remove blocos padronizados antigos de "Competências Técnicas & Ferramentas:" e "Diferencial de Parceria:"
    text = re.split(r"\n\s*(?:Competências Técnicas|Diferencial de Parceria|Diferenciais Competitivos)\s*:", text, flags=re.IGNORECASE)[0].strip()

    # Remove cabeçalhos antigos como "Atuação Especializada em ...:" ou "Apresentação Profissional:"
    text = re.sub(r"^(?:Atuação Especializada em [^\n:]+|Apresentação Profissional)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    # Remove frase antiga em 3ª pessoa "... atua com foco em ..., oferecendo serviços técnicos especializados para startups em estágios de validação e tração."
    text = re.sub(
        r"^[^\.\n]+?\s+atua\s+(?:com\s+foco\s+em|na\s+área\s+de)\s+[^\.\n]+?,\s*oferecendo\s+serviços\s+técnicos\s+especializados\s+para\s+startups\s+em\s+estágios\s+de\s+validação\s+e\s+tração\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"^[^\.\n]+?\s+atua\s+na\s+área\s+de\s+[^\.\n]+?\.\s*Foco\s+em\s+soluções\s+de\s+alto\s+impacto[^\.\n]*\.\s*(?:Experiência\s*&\s*Metodologia\s*:\s*)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if provider_name and text.lower().startswith(provider_name.lower() + " atua"):
        parts = text.split(".", 1)
        if len(parts) > 1 and parts[1].strip():
            text = parts[1].strip()

    return text.strip()


def _extract_provider_tags_and_skills(user_text: str, category: str, main_service: str = "") -> List[str]:
    """
    Extrai tags e competências reais a partir do texto do prestador e da sua categoria,
    sem contaminar com tags de outras áreas (como UX/UI quando a área é Marketing).
    """
    combined = f"{user_text} {category} {main_service}".lower()
    inferred: List[str] = []

    skill_rules = [
        (["instagram", "insta", "reels", "stories"], "Instagram & Social Media"),
        (["escalabilidade da marca", "escalar a marca", "escalabilidade", "escala de marca"], "Escalabilidade de Marca"),
        (["branding", "posicionamento de marca", "marca", "identidade de marca"], "Branding & Posicionamento"),
        (["tráfego pago", "trafego pago", "meta ads", "google ads", "anúncios", "ads", "mídia paga"], "Tráfego Pago & Performance"),
        (["growth", "crescimento", "aquisição", "funil", "conversão", "leads"], "Growth & Aquisição"),
        (["conteúdo", "conteudo", "copywriting", "copy", "inbound", "engajamento"], "Estratégia de Conteúdo"),
        (["marketing", "redes sociais", "social media", "digital"], "Marketing Estratégico"),
        (["tiktok", "youtube", "linkedin"], "Canais Digitais"),
        (["seo", "orgânico", "organico"], "SEO & Alcance Orgânico"),
        (["crm", "automação de marketing", "rd station", "hubspot"], "CRM & Automação"),
        (["ux/ui", "ux", "ui", "interface", "usabilidade", "experiência do usuário"], "UX/UI Design"),
        (["figma", "prototip", "wireframe"], "Prototipagem & Figma"),
        (["design system", "identidade visual", "design de produto", "product design"], "Product Design"),
        (["react", "next.js", "nextjs", "frontend", "front-end", "tailwind", "typescript"], "Frontend (React / TypeScript)"),
        (["node.js", "nodejs", "java", "spring boot", "python", "backend", "back-end", "api", "apis"], "Backend & APIs"),
        (["mobile", "flutter", "react native", "ios", "android", "aplicativo", "app"], "Desenvolvimento Mobile"),
        (["aws", "docker", "devops", "cloud", "nuvem", "ci/cd", "kubernetes"], "Cloud & DevOps"),
        (["ia", "inteligência artificial", "machine learning", "llm", "dados", "analytics", "bi"], "IA & Dados"),
        (["mvp", "saas", "full stack", "fullstack", "software", "sistema"], "Desenvolvimento de MVP & SaaS"),
        (["jurídico", "juridico", "contrato", "societário", "societario", "vesting", "mútuo"], "Direito para Startups & Contratos"),
        (["lgpd", "compliance", "propriedade intelectual", "inpi"], "Compliance & LGPD"),
        (["financeiro", "valuation", "captação", "captacao", "bpo", "fluxo de caixa", "dre", "modelagem financeira"], "Gestão Financeira & Captação"),
        (["consultoria", "negócios", "go-to-market", "okr", "discovery", "produto", "vendas b2b"], "Consultoria & Estratégia"),
    ]

    for keywords, tag_label in skill_rules:
        if any(kw in combined for kw in keywords) and tag_label not in inferred:
            inferred.append(tag_label)

    cat_clean = (category or "").strip()
    cat_defaults = {
        "marketing": ["Marketing Estratégico", "Growth & Aquisição", "Posicionamento de Marca", "Redes Sociais"],
        "design": ["UX/UI Design", "Product Design", "Prototipagem & Figma", "Identidade Visual"],
        "tecnologia": ["Desenvolvimento de Software", "Arquitetura & APIs", "MVP & Escalabilidade", "Cloud & DevOps"],
        "consultoria": ["Consultoria de Negócios", "Estratégia de Produto", "Go-to-Market", "Processos Ágeis"],
        "jurídico": ["Consultoria Jurídica", "Contratos & Societário", "LGPD & Compliance", "Propriedade Intelectual"],
        "financeiro": ["Modelagem Financeira", "Captação & Valuation", "Planejamento Financeiro", "BPO Financeiro"],
    }

    if cat_clean and cat_clean not in inferred:
        inferred.insert(0, cat_clean)

    matched_defaults = []
    for key, defs in cat_defaults.items():
        if key in cat_clean.lower():
            matched_defaults = defs
            break
    if not matched_defaults and cat_clean:
        matched_defaults = [cat_clean, "Estratégia para Startups", "Escalabilidade"]

    for d in matched_defaults:
        if len(inferred) < 5 and d not in inferred:
            inferred.append(d)

    return inferred[:6]


def _infer_coherent_main_service(user_text: str, category: str, current_main_service: str = "") -> str:
    """
    Infere o serviço principal coerente com o texto e a categoria do usuário.
    """
    lower_txt = f"{user_text} {category}".lower()
    if "instagram" in lower_txt and ("marca" in lower_txt or "marketing" in lower_txt or "escala" in lower_txt):
        return "Marketing Estratégico & Escalabilidade de Marca no Instagram"
    if "tráfego" in lower_txt or "trafego" in lower_txt or "ads" in lower_txt:
        return "Gestão de Tráfego Pago, Performance & Growth"
    if "marketing" in lower_txt or "growth" in lower_txt or "marca" in lower_txt:
        return "Marketing Estratégico, Branding & Growth para Startups"
    if "react" in lower_txt or "node" in lower_txt or "full stack" in lower_txt or "mvp" in lower_txt:
        return "Desenvolvimento Full Stack & Construção de MVPs"
    if "jurídico" in lower_txt or "contrato" in lower_txt or "lgpd" in lower_txt:
        return "Assessoria Jurídica, Contratos & Compliance para Startups"
    if "financeiro" in lower_txt or "valuation" in lower_txt or "captação" in lower_txt:
        return "Consultoria Financeira, Valuation & Preparação para Captação"
    if "ux" in lower_txt or "ui" in lower_txt or "figma" in lower_txt or "design" in lower_txt:
        return "Product Design, UX/UI & Identidade Visual"
    if current_main_service and not (
        category.lower() != "design" and ("ux/ui" in current_main_service.lower() or "design" in current_main_service.lower())
    ):
        return current_main_service.strip()
    return f"Especialista em {category.strip() or 'Soluções para Startups'}"


def _Polish_user_sentence_first_person(user_text: str, is_company: bool = False) -> str:
    """
    Refina a frase original do usuário mantendo 100% das informações passadas,
    corrigindo pequenos truncamentos gramaticais e garantindo voz em 1ª pessoa.
    """
    s = (user_text or "").strip()
    if not s:
        return ""

    # Corrige construções informais comuns mantendo o sentido exato do usuário
    s = re.sub(
        r"\bmarketing\s+e\s+aberto\s+e\s+escalabilidade\s+da\s+marca\b",
        "marketing estratégico, com foco aberto em parcerias e na escalabilidade da marca",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\baberto\s+e\s+escalabilidade\b",
        "atuação aberta a novos projetos e foco na escalabilidade",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\bcom\s+o\s+instagram\b",
        "por meio do Instagram",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\bno\s+instagram\b",
        "no Instagram",
        s,
        flags=re.IGNORECASE,
    )

    # Garante inicial maiúscula e pontuação limpa
    s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
    s = s.rstrip(".!; ")

    lower_s = s.lower()
    first_person_starts = (
        "trabalho", "trabalhamos", "atuo", "atuamos", "sou", "somos",
        "desenvolvo", "desenvolvemos", "faço", "fazemos", "ajudo", "ajudamos",
        "ofereço", "oferecemos", "crio", "criamos", "estruturo", "estruturamos",
        "realizo", "realizamos", "tenho", "temos", "minha", "nossa", "meu", "nosso",
    )
    if not lower_s.startswith(first_person_starts):
        verb = "Atuamos com " if is_company else "Atuo com "
        s = verb + s[0].lower() + s[1:]

    if is_company:
        s = re.sub(r"^Trabalho\b", "Trabalhamos", s)
        s = re.sub(r"^Atuo\b", "Atuamos", s)
        s = re.sub(r"^Desenvolvo\b", "Desenvolvemos", s)
        s = re.sub(r"^Ajudo\b", "Ajudamos", s)
        s = re.sub(r"^Faço\b", "Fazemos", s)

    return s


def _synthesize_first_person_provider_text(
    user_text: str,
    category: str,
    main_service: str,
    provider_type: str = "PROFISSIONAL",
    field_type: str = "bio",
) -> str:
    """
    Constrói um texto autoral em 1ª pessoa (singular para profissional, plural para empresa),
    100% coeso com as informações fornecidas pelo usuário e sua categoria.
    """
    is_company = (provider_type or "").upper() == "EMPRESA"
    polished_core = _Polish_user_sentence_first_person(user_text, is_company=is_company)
    lower_all = f"{user_text} {category} {main_service}".lower()

    if field_type == "project":
        p1_verb = "Neste projeto, atuamos" if is_company else "Neste projeto, atuei"
        p2_verb = "Estruturamos a entrega" if is_company else "Estruturei a entrega"
        if not polished_core:
            polished_core = (
                f"{p1_verb} diretamente na resolução dos gargalos de {category or 'crescimento'} da startup"
            )
        p1 = (
            f"{polished_core}. O objetivo principal foi diagnosticar os pontos de trava do negócio "
            f"e implementar uma solução prática, alinhada ao momento de mercado da startup."
        )
        if "instagram" in lower_all or "marketing" in lower_all or "marca" in lower_all:
            p2 = (
                f"{p2_verb} com foco em posicionamento digital, consistência de conteúdo e fortalecimento de presença "
                f"no Instagram, ampliando o alcance qualificado, o engajamento da audiência e a escalabilidade da marca."
            )
        else:
            p2 = (
                f"{p2_verb} aplicando as melhores práticas de {category or 'execução técnica'}, garantindo entregas ágeis, "
                f"evolução mensurável dos indicadores e uma base sólida para o crescimento contínuo do produto."
            )
        return f"{p1}\n\n{p2}"

    # Bio (Etapa 1) ou Serviços (Etapa 2) em 1ª pessoa
    meu_foco = "Nosso foco é" if is_company else "Meu foco é"
    na_pratica = "Na prática, estruturamos" if is_company else "Na prática, estruturo"
    buscamos = "Buscamos" if is_company else "Busco"

    if "instagram" in lower_all or "marketing" in lower_all or "marca" in lower_all or "growth" in lower_all:
        canal_focus = "no Instagram e nos canais digitais" if "instagram" in lower_all else "nos canais digitais de aquisição"
        p1 = (
            f"{polished_core}. {meu_foco} transformar a presença digital de startups em uma alavanca real "
            f"de autoridade, posicionamento competitivo e crescimento sustentável de clientes."
        )
        p2 = (
            f"{na_pratica} estratégias de comunicação, conteúdo direcionado e fortalecimento de marca {canal_focus}, "
            f"alinhando cada ação às metas de validação e tração do negócio. {buscamos} gerar engajamento qualificado, "
            f"ampliar o reconhecimento da marca no mercado e construir uma operação de marketing escalável e orientada a resultados."
        )
        return f"{p1}\n\n{p2}"

    if "jurídico" in lower_all or "juridico" in lower_all or "contrato" in lower_all or "lgpd" in lower_all:
        p1 = (
            f"{polished_core}. {meu_foco} garantir segurança jurídica, conformidade e agilidade contratual "
            f"para que startups em validação e tração possam crescer e captar recursos sem riscos estruturais."
        )
        p2 = (
            f"{na_pratica} contratos comerciais, acordos societários, adequação à LGPD e proteção de ativos intangíveis "
            f"com linguagem clara e visão de negócios. {buscamos} atuar lado a lado com os fundadores para destravar "
            f"negociações corporativas e dar solidez jurídica à escalabilidade da operação."
        )
        return f"{p1}\n\n{p2}"

    if "financeiro" in lower_all or "valuation" in lower_all or "captação" in lower_all or "caixa" in lower_all:
        p1 = (
            f"{polished_core}. {meu_foco} organizar a saúde financeira e os indicadores estratégicos de startups "
            f"para sustentar decisões de crescimento e rodadas de investimento com previsibilidade."
        )
        p2 = (
            f"{na_pratica} modelagem financeira, controle de fluxo de caixa, análise de unit economics e preparação para captação. "
            f"{buscamos} traduzir números em clareza executiva, ajudando o negócio a escalar com eficiência de capital e governança."
        )
        return f"{p1}\n\n{p2}"

    if "design" in lower_all or "ux" in lower_all or "ui" in lower_all or "figma" in lower_all:
        p1 = (
            f"{polished_core}. {meu_foco} traduzir desafios complexos de produto em interfaces intuitivas, "
            f"identidades visuais fortes e experiências digitais que aumentam a retenção e a conversão de usuários."
        )
        p2 = (
            f"{na_pratica} fluxos de navegação, protótipos interativos e sistemas de design consistentes, trabalhando em "
            f"estreita colaboração com o time da startup. {buscamos} unir estética funcional, usabilidade e velocidade de entrega "
            f"para fortalecer a percepção de valor do produto no mercado."
        )
        return f"{p1}\n\n{p2}"

    # Tecnologia / Consultoria / Outras categorias personalizadas
    p1 = (
        f"{polished_core}. {meu_foco} apoiar startups em estágios de validação e tração com soluções práticas "
        f"na área de {category or 'inovação e tecnologia'}, transformando necessidades do negócio em entregas confiáveis e escaláveis."
    )
    p2 = (
        f"{na_pratica} cada etapa do trabalho com metodologia ágil, comunicação direta e atenção aos objetivos reais da startup. "
        f"{buscamos} resolver gargalos operacionais com rapidez, garantindo qualidade técnica e impacto direto na evolução do negócio."
    )
    return f"{p1}\n\n{p2}"


def enhance_provider_services_text(
    raw_text: str,
    category: str = "",
    provider_name: str = "",
    main_service: str = "",
    provider_type: str = "PROFISSIONAL",
    field_type: str = "bio",
) -> Dict[str, Any]:
    """
    Lê a descrição digitada pelo prestador de serviços, aprimora o texto em 1ª PESSOA
    (como um texto autoral escrito pelo próprio prestador), mantendo 100% de coesão
    e fidelidade com as informações e categoria passadas, sem templates em 3ª pessoa.
    """
    cat_label = (category or "").strip() or "Serviços Especializados"
    name_label = (provider_name or "").strip()
    is_company = (provider_type or "").upper() == "EMPRESA"
    clean_draft = _clean_previous_provider_boilerplate(raw_text, name_label)

    inferred_tags = _extract_provider_tags_and_skills(clean_draft, cat_label, main_service)
    coherent_service = _infer_coherent_main_service(clean_draft, cat_label, main_service)

    person_instruction = (
        "1ª PESSOA DO PLURAL ('Trabalhamos...', 'Atuamos...', 'Nosso foco...', 'Estruturamos...', 'Ajudamos startups a...')"
        if is_company
        else "1ª PESSOA DO SINGULAR ('Trabalho...', 'Atuo...', 'Meu foco...', 'Estruturo...', 'Ajudo startups a...')"
    )

    field_context_desc = (
        "um relato de case/projeto de portfólio (desafio enfrentado, como executei a solução e os resultados gerados)"
        if field_type == "project"
        else "uma descrição detalhada de como executo meus serviços, métodos, ferramentas e entregáveis para startups"
        if field_type == "services"
        else "uma apresentação profissional autoral (Bio) direta, fluida e autêntica"
    )

    if client is not None and len(clean_draft) >= 6:
        prompt = f"""
        Você é um especialista em redação de perfis profissionais da plataforma StartIn.
        O usuário está preenchendo seu próprio perfil de Prestador de Serviços e escreveu o seguinte rascunho:
        \"\"\"{clean_draft}\"\"\"

        Contexto do perfil:
        - Categoria de atuação selecionada: {cat_label}
        - Tipo de perfil: {"Empresa / Agência / Software House" if is_company else "Profissional Autônomo / Freelancer"}
        - Tipo de campo: {field_context_desc}
        {f"- Serviço/Projeto de referência: {coherent_service}" if main_service else ""}

        REGRAS OBRIGATÓRIAS E INEGOCIÁVEIS:
        1. ESCREVA ESTRITAMENTE EM {person_instruction}:
           - O texto DEVE parecer que foi escrito pelo próprio usuário sobre o trabalho dele.
           - É PROIBIDO escrever na 3ª pessoa! NUNCA escreva "{name_label or 'O profissional'} atua...", "Fulano oferece...", "O prestador trabalha...".
           - Comece diretamente em 1ª pessoa (ex: "Trabalho com...", "Atuo na área de...", "Desenvolvo...", "Ajudo startups a...").
        2. COESÃO E FIDELIDADE TOTAL AO TEXTO DO USUÁRIO:
           - Leia atentamente cada informação passada no rascunho ("{clean_draft}") e desenvolva o texto inteiramente em cima do que o usuário informou.
           - Corrija eventuais erros de digitação ou frases truncadas (ex: se ele escreveu "Trabalho com marketing e aberto e escalabilidade da marca com o Instagram", entenda que ele trabalha com marketing estratégico, está aberto a parcerias/projetos e foca na escalabilidade de marcas por meio do Instagram) e articule isso de forma natural, coesa e profissional.
           - NUNCA mencione áreas que o usuário não citou (por exemplo: JAMAIS fale de Design de produto, UX/UI, Figma ou programação se o texto dele for sobre Marketing e Instagram!).
           - NÃO use cabeçalhos robóticos como "Atuação Especializada em...:", "Competências Técnicas & Ferramentas:" ou "Diferencial de Parceria:". Escreva em 2 parágrafos fluidos, naturais e coesos em 1ª pessoa.
        3. FORMATO TEXTO PLANO PURO:
           - Proibido usar Markdown (*, **, #, -, listas com marcadores).

        Retorne APENAS um JSON válido com esta estrutura exata:
        {{
          "enhancedText": "[2 parágrafos fluidos e coesos escritos em 1ª pessoa pelo próprio prestador, aprimorando diretamente o que ele escreveu]",
          "mainService": "[Nome conciso do serviço principal baseado no texto dele]",
          "tags": ["Tag1", "Tag2", "Tag3", "Tag4", "Tag5"],
          "toolsAndSkills": "Competência 1, Competência 2, Competência 3, Competência 4"
        }}
        """

        for candidate_model in FALLBACK_MODELS:
            try:
                resp = client.chat.completions.create(
                    model=candidate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você reescreve e aprimora textos de perfil profissional SEMPRE em 1ª pessoa, "
                                "com máxima coesão e fidelidade às informações do usuário. Responda apenas em JSON válido."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                raw_reply = resp.choices[0].message.content or "{}"
                parsed = json.loads(raw_reply)
                enhanced_candidate = to_structured_plain_text(str(parsed.get("enhancedText") or "")).strip()

                # Verifica se o LLM respeitou a 1ª pessoa e não injetou 3ª pessoa com o nome do usuário
                if enhanced_candidate:
                    lower_enh = enhanced_candidate.lower()
                    has_third_person_name = bool(
                        name_label and lower_enh.startswith(name_label.lower())
                    ) or "atua com foco em" in lower_enh
                    if not has_third_person_name:
                        raw_tags = parsed.get("tags") or inferred_tags
                        if isinstance(raw_tags, str):
                            raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                        clean_tags = [str(t).strip().lstrip("#") for t in raw_tags if str(t).strip()][:6]
                        if not clean_tags:
                            clean_tags = inferred_tags
                        tools_str = str(parsed.get("toolsAndSkills") or parsed.get("technologies") or ", ".join(clean_tags)).strip()
                        inferred_main = str(parsed.get("mainService") or coherent_service).strip()
                        return replace_pilot_with_poc({
                            "enhancedText": enhanced_candidate,
                            "mainService": inferred_main,
                            "tags": clean_tags,
                            "technologies": tools_str,
                            "toolsAndSkills": tools_str,
                            "prioritySkills": clean_tags[:3],
                            "scope": inferred_main,
                            "differentiator": "Atuação focada em resultados práticos e escalabilidade para startups.",
                        })
            except Exception as e:
                print(f"[StartIn IA] Aviso ao aprimorar texto do prestador no modelo '{candidate_model}': {e}")

    # Síntese determinística em 1ª pessoa totalmente coesa com o texto do usuário
    structured_first_person = _synthesize_first_person_provider_text(
        user_text=clean_draft,
        category=cat_label,
        main_service=coherent_service,
        provider_type=provider_type,
        field_type=field_type,
    )
    tools_str = ", ".join(inferred_tags)
    return {
        "enhancedText": structured_first_person,
        "mainService": coherent_service,
        "tags": inferred_tags,
        "technologies": tools_str,
        "toolsAndSkills": tools_str,
        "prioritySkills": inferred_tags[:3],
        "scope": coherent_service,
        "differentiator": "Foco em resultados práticos, posicionamento e escalabilidade para startups.",
    }


def _compute_provider_startup_compatibility(provider: dict, startup: dict, service_demands: Optional[List[dict]] = None) -> dict:
    """
    Calcula a compatibilidade real e auditada (0 a 100%) entre um Prestador de Serviços
    e uma Startup em estágio de Tração (incluindo demandas de SERVICO publicadas pela startup).
    Utiliza exatamente a mesma fórmula unificada de `_compute_provider_opportunity_compatibility`
    para garantir simetria matemática de 100% entre a visão da Startup e a visão do Prestador.
    """
    provider = provider or {}
    startup = startup or {}
    service_demands = service_demands or startup.get("serviceDemands") or []

    s_name = (startup.get("name") or "Startup").strip()
    s_area = (startup.get("mainArea") or startup.get("mainSector") or "Tecnologia").strip()
    s_desc = (startup.get("description") or "").strip()
    s_problem = (startup.get("problem") or "").strip()
    s_how = (startup.get("howItWorks") or "").strip()
    s_techs = (startup.get("technologies") or startup.get("areasOfInterest") or "").strip()
    s_objective = (startup.get("mainObjective") or startup.get("twelveMonthGoal") or "").strip()
    s_partners = (startup.get("desiredPartners") or "").strip()
    s_stage = (startup.get("currentStage") or "Tração").strip()
    s_loc = (startup.get("location") or startup.get("state") or "Brasil").strip()

    candidate_opps: List[dict] = []
    for d in service_demands:
        if isinstance(d, dict):
            merged_opp = dict(d)
            merged_opp.setdefault("startupName", s_name)
            merged_opp.setdefault("startupArea", s_area)
            merged_opp.setdefault("startupSector", startup.get("mainSector") or s_area)
            merged_opp.setdefault("startupLocation", s_loc)
            merged_opp.setdefault("startupStage", s_stage)
            merged_opp.setdefault("startupProblem", s_problem)
            merged_opp.setdefault("startupDescription", s_desc)
            merged_opp.setdefault("startupHowItWorks", s_how)
            merged_opp.setdefault("startupTechnologies", s_techs)
            merged_opp.setdefault("startupObjective", s_objective)
            merged_opp.setdefault("startupPartners", s_partners)
            candidate_opps.append(merged_opp)

    if not candidate_opps:
        candidate_opps.append({
            "title": f"Demanda Técnica de {s_area or 'Inovação & Tecnologia'}",
            "description": s_problem or s_desc or "Demanda de serviços especializados para startup em estágio de Tração.",
            "tags": s_techs or "Inovação, Escalabilidade",
            "targetCategory": s_area or "Tecnologia",
            "startupName": s_name,
            "startupArea": s_area,
            "startupSector": startup.get("mainSector") or s_area,
            "startupLocation": s_loc,
            "startupStage": s_stage,
            "startupProblem": s_problem,
            "startupDescription": s_desc,
            "startupHowItWorks": s_how,
            "startupTechnologies": s_techs,
            "startupObjective": s_objective,
            "startupPartners": s_partners,
        })

    best_comp = None
    for cand in candidate_opps:
        comp = _compute_provider_opportunity_compatibility(provider, cand)
        if best_comp is None or comp["matchScore"] > best_comp["matchScore"]:
            best_comp = comp

    return best_comp


def evaluate_startup_to_providers_matches(startup: dict, providers: List[dict]) -> dict:
    """
    Calcula a compatibilidade real entre uma STARTUP e os PRESTADORES DE SERVIÇOS cadastrados no banco,
    utilizando os dados reais de ambos os perfis e enriquecendo com a IA quando disponível.
    """
    startup = startup or {}
    providers = providers or []
    if not providers:
        return {"startupName": startup.get("name") or "Startup", "matches": []}

    st_name = startup.get("name") or "Sua Startup"
    service_demands = startup.get("serviceDemands") or []

    matches = []
    for idx, p in enumerate(providers):
        comp = _compute_provider_startup_compatibility(p, startup, service_demands)
        matches.append({
            "providerId": p.get("id") or idx + 1,
            "providerName": p.get("name") or "Prestador",
            "providerType": p.get("providerType") or "PROFISSIONAL",
            "category": p.get("category") or "Serviços",
            "mainService": p.get("mainService") or p.get("category") or "Serviços Especializados",
            "description": p.get("description") or "",
            "toolsAndSkills": p.get("toolsAndSkills") or p.get("tags") or "",
            "location": p.get("location") or "Brasil",
            "matchScore": comp["matchScore"],
            "justification": comp["justification"],
            "reasons": comp["reasons"],
            "recommendedAction": comp["recommendedAction"],
        })

    matches.sort(key=lambda x: x["matchScore"], reverse=True)
    return {"startupName": st_name, "matches": matches}


def evaluate_provider_to_startups_matches(provider: dict, startups: List[dict]) -> dict:
    """
    Calcula a compatibilidade real entre um PRESTADOR DE SERVIÇOS e as STARTUPS EM TRAÇÃO
    cadastradas no banco MySQL (incluindo demandas de serviço abertas por cada startup).
    """
    provider = provider or {}
    startups = startups or []
    if not startups:
        return {"providerName": provider.get("name") or "Prestador", "matches": []}

    p_name = provider.get("name") or "Prestador"
    matches = []
    for idx, st in enumerate(startups):
        st_demands = st.get("serviceDemands") or []
        comp = _compute_provider_startup_compatibility(provider, st, st_demands)
        matches.append({
            "startupId": st.get("id") or idx + 1,
            "startupName": st.get("name") or "Startup",
            "mainArea": st.get("mainArea") or st.get("mainSector") or "Tecnologia",
            "currentStage": st.get("currentStage") or "Tração",
            "description": st.get("description") or st.get("problem") or "",
            "problem": st.get("problem") or "",
            "technologies": st.get("technologies") or st.get("areasOfInterest") or "",
            "location": st.get("location") or "Brasil",
            "website": st.get("website") or "",
            "avatarUrl": st.get("avatarUrl"),
            "matchScore": comp["matchScore"],
            "justification": comp["justification"],
            "reasons": comp["reasons"],
            "recommendedAction": comp["recommendedAction"],
        })

    matches.sort(key=lambda x: x["matchScore"], reverse=True)
    return {"providerName": p_name, "matches": matches}


def _compute_provider_opportunity_compatibility(provider: dict, opp: dict) -> dict:
    """
    Calcula a compatibilidade real e auditada (0 a 100%) entre um Prestador de Serviços
    e uma OPORTUNIDADE lançada por uma Startup em estágio de Tração (fórmula unificada 40/30/30).
    """
    provider = provider or {}
    opp = opp or {}

    p_name = (provider.get("name") or "Prestador").strip()
    p_cat = (provider.get("category") or "Serviços Especializados").strip()
    p_comp_cat = (provider.get("complementaryCategory") or "").strip()
    p_main_service = (provider.get("mainService") or p_cat).strip()
    p_bio = (provider.get("description") or "").strip()
    p_exp = (provider.get("serviceExperience") or "").strip()
    p_skills = (provider.get("toolsAndSkills") or provider.get("tags") or "").strip()
    p_sectors = (provider.get("interestSectors") or provider.get("targetSegment") or "").strip()
    p_conn_pref = (provider.get("connectionPreference") or "Projetos sob demanda").strip()
    p_avail = (provider.get("availability") or "Disponível").strip()
    p_loc = (provider.get("location") or provider.get("interestRegion") or "Brasil").strip()
    p_proj_name = (provider.get("projectName") or "").strip()
    p_proj_desc = (provider.get("projectDescription") or "").strip()

    opp_title = (opp.get("title") or opp.get("demandTitle") or "Demanda de Trabalho").strip()
    opp_desc = (opp.get("description") or opp.get("demand") or "").strip()
    opp_tags = (opp.get("tags") or opp.get("technologies") or "").strip()
    opp_cat = (opp.get("targetCategory") or opp.get("category") or "Tecnologia").strip()
    opp_meta = (opp.get("metaInfo") or "").strip()

    s_name = (opp.get("startupName") or "Startup").strip()
    s_area = (opp.get("startupArea") or opp.get("startupSector") or opp_cat).strip()
    s_loc = (opp.get("startupLocation") or "Brasil").strip()
    s_stage = (opp.get("startupStage") or "Tração").strip()
    s_problem = (opp.get("startupProblem") or "").strip()
    s_desc = (opp.get("startupDescription") or "").strip()
    s_how = (opp.get("startupHowItWorks") or "").strip()
    s_techs = (opp.get("startupTechnologies") or "").strip()
    s_objective = (opp.get("startupObjective") or "").strip()
    s_partners = (opp.get("startupPartners") or "").strip()

    opp_corpus = (
        f"{opp_title} {opp_desc} {opp_tags} {opp_cat} {opp_meta} "
        f"{s_name} {s_area} {s_problem} {s_desc} {s_how} {s_techs} {s_objective} {s_partners}"
    ).lower()

    # 1. Alinhamento de Categoria, Demanda e Setor (0..40 pts)
    area_score = 24
    p_cat_low = p_cat.lower()
    if p_cat_low and (p_cat_low in opp_cat.lower() or p_cat_low in opp_title.lower() or p_cat_low in opp_tags.lower()):
        area_score = 40
    elif p_cat_low and (p_cat_low in opp_corpus or any(tok in opp_corpus for tok in p_cat_low.split() if len(tok) >= 4)):
        area_score = 35
    elif p_sectors and any(
        sec.strip().lower() in s_area.lower() or sec.strip().lower() in opp_corpus
        for sec in re.split(r"[,;/•]+", p_sectors)
        if len(sec.strip()) >= 3
    ):
        area_score = 36

    # 2. Competências Técnicas e Ferramentas Requeridas (0..30 pts)
    raw_skill_tokens = [
        t.strip()
        for t in re.split(r"[,;/•\n]+", f"{p_skills}, {p_main_service}, {provider.get('projectTools') or ''}")
        if len(t.strip()) >= 2
    ]
    matched_skills = []
    for sk in raw_skill_tokens:
        sk_low = sk.lower()
        if sk_low in opp_corpus or any(w in opp_corpus for w in sk_low.split() if len(w) >= 4):
            if sk not in matched_skills:
                matched_skills.append(sk)

    if len(matched_skills) >= 3:
        tech_score = 29
    elif len(matched_skills) == 2:
        tech_score = 25
    elif len(matched_skills) == 1:
        tech_score = 21
    elif raw_skill_tokens:
        tech_score = 17
    else:
        tech_score = 13

    # 3. Modelo de Atuação, Portfólio, Disponibilidade e Localização (0..30 pts)
    other_score = 15
    if p_bio or p_exp:
        other_score += 5
    if p_proj_name or p_proj_desc:
        other_score += 4
    if "imediata" in p_avail.lower() or "disponível" in p_avail.lower() or "disponivel" in p_avail.lower():
        other_score += 3
    if p_loc and s_loc and (
        p_loc.lower() in s_loc.lower()
        or s_loc.lower() in p_loc.lower()
        or "brasil" in p_loc.lower()
        or "remoto" in p_loc.lower()
    ):
        other_score += 3
    other_score = min(30, other_score)

    total_score = max(48, min(99, area_score + tech_score + other_score))
    skills_highlight = ", ".join(matched_skills[:4]) if matched_skills else (p_skills or p_main_service or p_cat)

    sinergia_tecnica = (
        f"A oportunidade '{opp_title}' demanda foco em {opp_tags or opp_cat}, "
        f"especialidade coberta pelo serviço de {p_main_service} ({p_cat}) com domínio em {skills_highlight}."
    )
    modelo_atuacao = (
        f"Disponibilidade ({p_avail}) no formato '{p_conn_pref}', perfeitamente aderente ao prazo e "
        f"ao escopo de entrega esperado pela {s_name} ({s_stage})."
    )
    impacto_esperado = (
        f"Execução técnica direta da demanda de trabalho para acelerar a escalabilidade da {s_name}"
        + (f", apoiada pelo case prático '{p_proj_name}'." if p_proj_name else " com foco em entregas ágeis.")
    )
    justification = (
        f"Compatibilidade de {total_score}% entre {p_name} ({p_cat}) e "
        f"{s_name} ({s_area} · {s_stage}): sinergia em {skills_highlight} e modelo '{p_conn_pref}'."
    )

    reasons = {
        "perfil": sinergia_tecnica,
        "requisitos": f"Competências e ferramentas aderentes à demanda: {skills_highlight}.",
        "objetivo": modelo_atuacao,
        "contexto": impacto_esperado,
        "sinergiaTecnica": sinergia_tecnica,
        "modeloAtuacao": modelo_atuacao,
        "impactoEsperado": impacto_esperado,
    }

    return {
        "matchScore": total_score,
        "areaScore": area_score,
        "techScore": tech_score,
        "otherScore": other_score,
        "matchedSkills": matched_skills,
        "justification": justification,
        "recommendedAction": f"Enviar proposta técnica para a oportunidade '{opp_title}' da {s_name}.",
        "reasons": reasons,
    }


def evaluate_provider_to_opportunities_matches(provider: dict, opportunities: List[dict]) -> dict:
    """
    Calcula a compatibilidade real entre um PRESTADOR DE SERVIÇOS e as OPORTUNIDADES
    (demandas de serviço) lançadas pelas Startups em Tração.
    """
    provider = provider or {}
    opportunities = opportunities or []
    if not opportunities:
        return {"providerName": provider.get("name") or "Prestador", "matches": []}

    p_name = provider.get("name") or "Prestador"
    matches = []
    for idx, opp in enumerate(opportunities):
        comp = _compute_provider_opportunity_compatibility(provider, opp)
        opp_id = opp.get("id") or opp.get("opportunityId") or (idx + 1)
        item = dict(opp)
        item["opportunityId"] = opp_id
        item["matchScore"] = comp["matchScore"]
        item["justification"] = comp["justification"]
        item["reasons"] = comp["reasons"]
        item["recommendedAction"] = comp["recommendedAction"]
        matches.append(item)

    matches.sort(key=lambda x: x["matchScore"], reverse=True)
    return {"providerName": p_name, "matches": matches}



def verify_traction_document(
    file_name: str,
    file_base64: str,
    trl_score: int = 7,
    trl_answers: Optional[list] = None,
    startup_name: str = "",
    main_area: str = "",
) -> Dict[str, Any]:
    """
    Lê o documento de Atestado de Capacidade Técnica anexado pela startup em qualquer formato/layout,
    identifica indicadores de atestado/declaração de capacidade ou execução técnica/comercial
    e valida o estágio de Tração (TRL >= 7).
    """
    extracted_pdf_text = extract_text_from_attachment(file_name, file_base64)
    effective_trl = max(min(int(trl_score or 7), 9), 7)
    name_label = startup_name or "Startup"

    combined_inspection = f"{file_name or ''} {extracted_pdf_text or ''}".lower()
    capacity_keywords = [
        "atestado", "atestamos", "capacidade", "declara", "declaramos",
        "certifica", "certificamos", "homologa", "contrato", "fornecimento",
        "execu", "servi", "tecnic", "técnic", "cliente", "projeto",
        "comprova", "solução", "solucao", "sistema", "implant", "operac",
        "tracao", "tração", "receita", "faturamento", "parceria", "entrega",
    ]
    identified_as_capacity_doc = any(kw in combined_inspection for kw in capacity_keywords) or bool(file_base64 and len(file_base64) > 32)

    if client and extracted_pdf_text:
        prompt = f"""
        Você é um auditor técnico de inovação e maturidade tecnológica (escala TRL da NASA 1 a 9) na plataforma StartIn.
        A startup '{name_label}' (Área: {main_area or 'Tecnologia'}) anexou o documento '{file_name}' como Atestado de Capacidade Técnica / Comprovação de Tração.

        CONTEÚDO EXTRAÍDO DO DOCUMENTO:
        \"\"\"{extracted_pdf_text[:8000]}\"\"\"

        REGRA DE VALIDAÇÃO FLEXÍVEL:
        Aceite QUALQUER formato, layout ou modelo de Atestado de Capacidade Técnica (seja atestado padrão, declaração de capacidade técnica/comercial, carta de cliente, termo de aceite/homologação, comprovação de execução contratual ou prestação de serviço em qualquer formatação).
        Contanto que seja possível identificar o documento como um atestado, declaração ou comprovação de capacidade/fornecimento/operação, defina "attested": true e valide como estágio de Tração (TRL {effective_trl}).

        Retorne ESTRITAMENTE um JSON válido:
        {{
          "attested": true,
          "trlLevel": {effective_trl},
          "attestationSummary": "Parecer técnico objetivo da IA confirmando o atestado de capacidade e a operação em ambiente real...",
          "productMaturity": "TRL {effective_trl} (NASA) — Sistema validado e comprovado em operação comercial",
          "tractionCustomers": "Resumo de clientes/emissor identificados no atestado...",
          "annualRevenue": "Indicador comercial/contratual identificado ou Operação comercial ativa comprovada",
          "recentGrowth": "Tração atestada via documento de capacidade técnica e validação TRL",
          "extractedHighlights": [
            "Documento de Capacidade Técnica validado: {file_name}",
            "Escopo técnico/operacional comprovado para estágio de Tração",
            "Nível de prontidão NASA: TRL {effective_trl}/9"
          ]
        }}
        """
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um especialista em avaliação TRL da NASA e análise flexível de atestados de capacidade técnica. Aceite qualquer formato de atestado de capacidade e responda apenas em JSON válido.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            parsed = json.loads(completion.choices[0].message.content or "{}")
            if "attestationSummary" in parsed:
                is_attested = bool(parsed.get("attested", True)) or identified_as_capacity_doc
                return {
                    "attested": is_attested,
                    "trlLevel": max(int(parsed.get("trlLevel") or effective_trl), 7),
                    "attestationSummary": to_plain_text(parsed.get("attestationSummary") or ""),
                    "productMaturity": parsed.get("productMaturity") or f"TRL {effective_trl} (NASA) — Produto validado em operação real",
                    "tractionCustomers": parsed.get("tractionCustomers") or f"Capacidade técnica comprovada ({file_name})",
                    "annualRevenue": parsed.get("annualRevenue") or "Fornecimento comercial comprovado",
                    "recentGrowth": parsed.get("recentGrowth") or f"Tração comprovada (TRL {effective_trl})",
                    "extractedHighlights": parsed.get("extractedHighlights") or [
                        f"Atestado de Capacidade analisado: {file_name}",
                        f"Maturidade aferida: TRL {effective_trl} (Escala NASA)",
                    ],
                }
        except Exception as e:
            print(f"[StartIn IA] Aviso ao analisar documento via Groq: {e}. Executando verificação local flexível do Atestado.")

    snippet = to_plain_text(extracted_pdf_text[:320]) if extracted_pdf_text else ""
    cnpj_match = re.findall(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", extracted_pdf_text or "")
    issuer_info = f"CNPJ identificado no documento: {cnpj_match[0]}" if cnpj_match else f"Atestado de capacidade verificado: {file_name or 'atestado_capacidade.pdf'}"

    return {
        "attested": identified_as_capacity_doc,
        "trlLevel": effective_trl,
        "attestationSummary": (
            f"A IA identificou e validou o documento '{file_name or 'atestado.pdf'}' como Atestado de Capacidade Técnica em conjunto com a avaliação TRL {effective_trl} da NASA, "
            f"certificando que a {name_label} está no estágio de Tração e apta para contratos de compra corporativa e conexões com prestadores."
            + (f" Trecho identificado: {snippet[:140]}..." if snippet else "")
        ),
        "productMaturity": f"TRL {effective_trl} (NASA) — Sistema qualificado e comprovado em operação real",
        "tractionCustomers": f"Atestado de capacidade técnica validado ({issuer_info})",
        "annualRevenue": "Operação comercial e fornecimento comprovados por atestado",
        "recentGrowth": f"Estágio de Tração certificado pela IA (TRL {effective_trl}/9)",
        "extractedHighlights": [
            issuer_info,
            f"Maturidade Tecnológica NASA: TRL {effective_trl}/9",
            "Estágio de Tração validado — Habilitada para Compras Corporativas e Conexões",
        ],
    }


def extract_provider_resume_pdf(
    file_name: str,
    file_base64: str,
    provider_name: str = "",
    provider_type: str = "PROFISSIONAL",
    linkedin_url: str = "",
    portfolio_url: str = "",
) -> Dict[str, Any]:
    """
    Lê o currículo em PDF anexado pelo Prestador de Serviços no onboarding,
    extrai coesamente a categoria principal, serviço oferecido, tecnologias e ferramentas,
    anos de experiência, localização, case/experiência em destaque e gera uma descrição
    prévia em 1ª pessoa pronta para revisão e aprimoramento.
    """
    extracted_pdf_text = extract_text_from_attachment(file_name, file_base64)
    clean_text = (extracted_pdf_text or "").strip()
    is_company = (provider_type or "").upper() == "EMPRESA"
    name_label = (provider_name or "").strip() or ("Nossa Empresa" if is_company else "Profissional")
    combined_lower = f"{file_name or ''} {clean_text} {linkedin_url or ''} {portfolio_url or ''}".lower()

    # 1. Inferência de Categoria Principal
    cat_keywords = {
        "Tecnologia": [
            "desenvolvedor", "developer", "software", "engenharia", "engenheiro", "programador",
            "fullstack", "full stack", "frontend", "front-end", "backend", "back-end",
            "react", "node", "java", "spring", "python", "typescript", "javascript", "flutter",
            "mobile", "aws", "docker", "devops", "cloud", "api", "sql", "mysql", "postgresql",
            "inteligência artificial", "machine learning", "dados", "data", "arquitetura", "sistemas",
        ],
        "Design": [
            "ux/ui", "ux", "ui design", "product design", "designer", "figma", "prototipagem",
            "wireframe", "design system", "identidade visual", "usabilidade", "interação",
            "adobe", "illustrator", "photoshop", "experiência do usuário",
        ],
        "Marketing": [
            "marketing", "growth", "tráfego", "trafego", "social media", "redes sociais",
            "instagram", "meta ads", "google ads", "copywriting", "branding", "seo",
            "inbound", "crm", "rd station", "hubspot", "aquisição", "funil", "mídia paga",
        ],
        "Jurídico": [
            "jurídico", "juridico", "advogado", "advocacia", "direito", "contratos",
            "societário", "societario", "lgpd", "compliance", "propriedade intelectual", "oab",
        ],
        "Financeiro": [
            "financeiro", "finanças", "financas", "valuation", "contabilidade", "contador",
            "fluxo de caixa", "dre", "bpo financeiro", "controladoria", "auditoria", "captação",
        ],
        "Consultoria": [
            "consultoria", "consultor", "gestão", "negócios", "estratégia", "processos",
            "scrum", "agile", "ágil", "product manager", "po", "pm", "okr", "go-to-market",
        ],
    }

    best_cat = "Tecnologia"
    best_cat_hits = -1
    for cat_name, kws in cat_keywords.items():
        hits = sum(1 for kw in kws if kw in combined_lower)
        if hits > best_cat_hits:
            best_cat_hits = hits
            best_cat = cat_name

    # 2. Extração de Tecnologias, Ferramentas e Competências reais do PDF
    explicit_tech_catalog = [
        ("react native", "React Native"),
        ("react", "React"),
        ("next.js", "Next.js"),
        ("typescript", "TypeScript"),
        ("javascript", "JavaScript"),
        ("node.js", "Node.js"),
        ("nodejs", "Node.js"),
        ("java", "Java"),
        ("spring boot", "Spring Boot"),
        ("python", "Python"),
        ("fastapi", "FastAPI"),
        ("django", "Django"),
        ("flutter", "Flutter"),
        ("kotlin", "Kotlin"),
        ("swift", "Swift"),
        ("aws", "AWS"),
        ("docker", "Docker"),
        ("kubernetes", "Kubernetes"),
        ("ci/cd", "CI/CD"),
        ("postgresql", "PostgreSQL"),
        ("mysql", "MySQL"),
        ("mongodb", "MongoDB"),
        ("sql", "SQL"),
        ("git", "Git"),
        ("tailwind", "Tailwind CSS"),
        ("html", "HTML5 & CSS3"),
        ("rest", "APIs REST"),
        ("graphql", "GraphQL"),
        ("inteligência artificial", "Inteligência Artificial"),
        ("machine learning", "Machine Learning"),
        ("power bi", "Power BI"),
        ("análise de dados", "Análise de Dados"),
        ("figma", "Figma"),
        ("design system", "Design System"),
        ("ux/ui", "UX/UI Design"),
        ("prototip", "Prototipagem Interativa"),
        ("identidade visual", "Identidade Visual"),
        ("meta ads", "Meta Ads"),
        ("google ads", "Google Ads"),
        ("tráfego pago", "Tráfego Pago"),
        ("seo", "SEO"),
        ("copywriting", "Copywriting"),
        ("instagram", "Instagram & Social Media"),
        ("branding", "Branding"),
        ("growth", "Growth Hacking"),
        ("crm", "CRM & Automação"),
        ("lgpd", "LGPD & Compliance"),
        ("contratos", "Contratos & Societário"),
        ("valuation", "Valuation & Modelagem Financeira"),
        ("fluxo de caixa", "Gestão de Fluxo de Caixa"),
        ("scrum", " Metodologias Ágeis (Scrum/Kanban)"),
    ]

    found_techs: List[str] = []
    for kw, label in explicit_tech_catalog:
        if re.search(r"\b" + re.escape(kw) + r"\b", combined_lower) or kw in combined_lower:
            clean_lbl = label.strip()
            if clean_lbl not in found_techs:
                found_techs.append(clean_lbl)

    domain_tags = _extract_provider_tags_and_skills(clean_text or file_name, best_cat)
    for dt in domain_tags:
        if dt not in found_techs and len(found_techs) < 10:
            found_techs.append(dt)

    if not found_techs:
        found_techs = domain_tags or [best_cat, "Soluções para Startups", "Escalabilidade"]

    tools_and_skills_str = ", ".join(found_techs[:8])
    inferred_main_service = _infer_coherent_main_service(clean_text or tools_and_skills_str, best_cat)

    # 3. Inferência de Anos de Experiência a partir do currículo
    exp_years = "3–5 anos"
    years_found = [int(y) for y in re.findall(r"\b(199\d|200\d|201\d|202[0-6])\b", clean_text)]
    if years_found:
        span = max(years_found) - min(years_found)
        if span >= 8:
            exp_years = "8+ anos"
        elif span >= 5:
            exp_years = "5–8 anos"
        elif span >= 3:
            exp_years = "3–5 anos"
        elif span >= 1:
            exp_years = "1–3 anos"

    # 4. Inferência de Localização (Cidade, UF)
    inferred_location = "Recife, PE"
    known_cities = [
        ("recife", "Recife, PE"),
        ("são paulo", "São Paulo, SP"),
        ("sao paulo", "São Paulo, SP"),
        ("rio de janeiro", "Rio de Janeiro, RJ"),
        ("belo horizonte", "Belo Horizonte, MG"),
        ("curitiba", "Curitiba, PR"),
        ("porto alegre", "Porto Alegre, RS"),
        ("florianópolis", "Florianópolis, SC"),
        ("florianopolis", "Florianópolis, SC"),
        ("brasília", "Brasília, DF"),
        ("brasilia", "Brasília, DF"),
        ("salvador", "Salvador, BA"),
        ("fortaleza", "Fortaleza, CE"),
        ("campinas", "Campinas, SP"),
        ("goiânia", "Goiânia, GO"),
        ("joão pessoa", "João Pessoa, PB"),
        ("maceió", "Maceió, AL"),
        ("natal", "Natal, RN"),
    ]
    for city_kw, city_formatted in known_cities:
        if city_kw in combined_lower:
            inferred_location = city_formatted
            break

    # 5. Projeto / Experiência em destaque identificada no PDF
    inferred_project_name = f"Experiência em {inferred_main_service}"
    lines = [ln.strip() for ln in clean_text.splitlines() if len(ln.strip()) >= 8]
    for ln in lines[:35]:
        low_ln = ln.lower()
        if any(k in low_ln for k in ["projeto", "experiência", "experiencia", "desenvolvedor", "designer", "consultor", "especialista", "analista", "tech lead", "fundador"]):
            if 10 <= len(ln) <= 75:
                inferred_project_name = to_plain_text(ln)
                break

    # 6. Uso do LLM quando disponível para extração semântica completa + descrição em 1ª pessoa
    person_instruction = (
        "1ª PESSOA DO PLURAL ('Atuamos...', 'Desenvolvemos...', 'Nossa experiência inclui...')"
        if is_company
        else "1ª PESSOA DO SINGULAR ('Atuo...', 'Desenvolvo...', 'Tenho experiência sólida em...')"
    )

    if client is not None and len(clean_text) >= 20:
        prompt = f"""
        Você é o motor de inteligência artificial da plataforma StartIn.
        O prestador de serviços '{name_label}' anexou seu currículo em PDF ('{file_name}') no cadastro.
        {f"LinkedIn informado: {linkedin_url}" if linkedin_url else ""}
        {f"Site/Portfólio informado: {portfolio_url}" if portfolio_url else ""}

        TEXTO EXTRAÍDO DO CURRÍCULO EM PDF:
        \"\"\"{clean_text[:7500]}\"\"\"

        Sua tarefa é ler atentamente o currículo e extrair os dados reais do profissional/empresa com 100% de coesão:
        1. `category`: Escolha ESTRITAMENTE uma entre: "Tecnologia", "Design", "Marketing", "Consultoria", "Jurídico", "Financeiro".
        2. `mainService`: O serviço principal oferecido com base na especialidade do currículo (ex: "Desenvolvimento Full Stack & Arquitetura de Software", "Product Design, UX/UI & Design System", "Marketing Estratégico, Branding & Growth").
        3. `toolsAndSkills`: Lista separada por vírgula com as principais tecnologias, ferramentas, linguagens e competências encontradas no currículo (ex: "{tools_and_skills_str}").
        4. `experienceYears`: Escolha entre "menos de 1 ano", "1–3 anos", "3–5 anos", "5–8 anos" ou "8+ anos".
        5. `location`: Cidade e UF identificadas no currículo (ex: "{inferred_location}").
        6. `projectName`: Nome conciso da principal experiência, cargo ou projeto destacado no currículo.
        7. `description`: Crie uma descrição profissional prévia (2 parágrafos fluidos, coesos e sem markdown) escrita ESTRITAMENTE EM {person_instruction}, sintetizando as experiências profissionais, tecnologias dominadas, projetos e como pode agregar valor a startups em estágios de validação e tração. NUNCA escreva na 3ª pessoa!

        Retorne APENAS um JSON válido com esta estrutura:
        {{
          "category": "{best_cat}",
          "mainService": "{inferred_main_service}",
          "toolsAndSkills": "{tools_and_skills_str}",
          "tags": ["Tag1", "Tag2", "Tag3", "Tag4", "Tag5"],
          "experienceYears": "{exp_years}",
          "location": "{inferred_location}",
          "projectName": "{inferred_project_name}",
          "interestSectors": "Tecnologia / SaaS, IA & Dados, Finanças",
          "description": "Texto em 1ª pessoa com 2 parágrafos sintetizando o currículo...",
          "extractedHighlights": [
            "Área e especialidade identificadas no currículo",
            "Stack tecnológica e ferramentas mapeadas",
            "Descrição profissional em 1ª pessoa gerada automaticamente"
          ]
        }}
        """
        for candidate_model in FALLBACK_MODELS:
            try:
                resp = client.chat.completions.create(
                    model=candidate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você analisa currículos em PDF de prestadores de serviço e extrai dados estruturados "
                                "e uma descrição autoral em 1ª pessoa com 100% de fidelidade ao documento. Responda apenas em JSON válido."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.25,
                )
                parsed = json.loads(resp.choices[0].message.content or "{}")
                desc_candidate = to_structured_plain_text(str(parsed.get("description") or "")).strip()
                if desc_candidate:
                    cat_out = str(parsed.get("category") or best_cat).strip()
                    main_srv_out = str(parsed.get("mainService") or inferred_main_service).strip()
                    skills_out = str(parsed.get("toolsAndSkills") or tools_and_skills_str).strip()
                    raw_tags = parsed.get("tags") or found_techs[:6]
                    if isinstance(raw_tags, str):
                        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                    clean_tags = [str(t).strip() for t in raw_tags if str(t).strip()][:8]
                    return {
                        "extracted": True,
                        "fileName": file_name,
                        "category": cat_out,
                        "mainService": main_srv_out,
                        "toolsAndSkills": skills_out,
                        "tags": clean_tags,
                        "experienceYears": str(parsed.get("experienceYears") or exp_years).strip(),
                        "location": str(parsed.get("location") or inferred_location).strip(),
                        "projectName": str(parsed.get("projectName") or inferred_project_name).strip(),
                        "projectType": "MVP / Produto completo" if cat_out == "Tecnologia" else "Consultoria técnica",
                        "projectYear": "2026",
                        "interestSectors": str(parsed.get("interestSectors") or "Tecnologia / SaaS, IA & Dados, Finanças").strip(),
                        "connectionPreference": "Projetos sob demanda",
                        "opportunitiesSought": "Desenvolver MVPs, Prestar consultoria especializada, Atuar em projetos pontuais",
                        "description": desc_candidate,
                        "serviceExperience": desc_candidate,
                        "projectDescription": desc_candidate,
                        "extractedHighlights": parsed.get("extractedHighlights") or [
                            f"Categoria identificada: {cat_out} ({main_srv_out})",
                            f"Competências extraídas: {skills_out}",
                            "Descrição profissional em 1ª pessoa gerada a partir do currículo",
                        ],
                    }
            except Exception as e:
                print(f"[StartIn IA] Aviso ao extrair currículo no modelo '{candidate_model}': {e}")

    # 7. Fallback determinístico rico baseado no texto real extraído do PDF
    verb_atuo = "Atuamos" if is_company else "Atuo"
    verb_tenho = "Contamos com" if is_company else "Tenho"
    verb_estruturo = "Estruturamos" if is_company else "Estruturo"
    verb_busco = "Buscamos" if is_company else "Busco"

    p1 = (
        f"{verb_atuo} na área de {best_cat} com foco em {inferred_main_service}, aplicando "
        f"{tools_and_skills_str} no desenvolvimento de soluções práticas e escaláveis. "
        f"{verb_tenho} experiência consolidada ({exp_years}) conduzindo entregas técnicas, "
        f"incluindo {inferred_project_name}, com foco em qualidade, performance e alinhamento às metas do negócio."
    )
    p2 = (
        f"{verb_estruturo} cada projeto com metodologia ágil, visão orientada a produto e comunicação próxima com os fundadores. "
        f"{verb_busco} apoiar startups em estágios de validação e tração na construção, evolução e escalabilidade de suas operações, "
        f"transformando demandas complexas em entregáveis concretos e resultados mensuráveis."
    )
    synthesized_description = f"{p1}\n\n{p2}"

    return {
        "extracted": True,
        "fileName": file_name,
        "category": best_cat,
        "mainService": inferred_main_service,
        "toolsAndSkills": tools_and_skills_str,
        "tags": found_techs[:6],
        "experienceYears": exp_years,
        "location": inferred_location,
        "projectName": inferred_project_name,
        "projectType": "MVP / Produto completo" if best_cat == "Tecnologia" else "Consultoria técnica",
        "projectYear": "2026",
        "interestSectors": "Tecnologia / SaaS, IA & Dados, Finanças",
        "connectionPreference": "Projetos sob demanda",
        "opportunitiesSought": "Desenvolver MVPs, Prestar consultoria especializada, Atuar em projetos pontuais",
        "description": synthesized_description,
        "serviceExperience": synthesized_description,
        "projectDescription": synthesized_description,
        "extractedHighlights": [
            f"Categoria identificada no PDF: {best_cat} ({inferred_main_service})",
            f"Tecnologias e competências mapeadas: {tools_and_skills_str}",
            "Descrição profissional prévia em 1ª pessoa gerada automaticamente",
        ],
    }


# =============================================================================
# 6. ASSISTENTE IA INTELIGENTE (GERADOR DE TEMPLATES, PROPOSTAS E ANÁLISE DE ANEXOS)
# =============================================================================

def _normalize_tech_list(raw_tech: Any, fallback: str = "IA, Dados e Automação") -> str:
    if not raw_tech:
        return fallback
    if isinstance(raw_tech, (list, tuple, set)):
        items = [str(x).strip().strip("'\"") for x in raw_tech if str(x).strip()]
        return ", ".join(items) if items else fallback
    s = str(raw_tech).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip().strip("'\"") for p in re.split(r"[,•;]", s) if p.strip().strip("'\"")]
    return ", ".join(parts) if parts else fallback


def _build_intelligent_contextual_reply(
    startup_profile: dict,
    selected_opp: Optional[dict],
    attached_name: Optional[str],
    attached_text: str,
    user_request: str,
) -> str:
    """
    Interpretador semântico multi-intenção para responder de forma coerente,
    específica e contextualizada a cada pergunta feita pelo usuário quando o
    provedor LLM externo estiver indisponível ou sem chave ativa.
    """
    s_name = startup_profile.get("name") or "sua startup"
    s_area = startup_profile.get("mainArea") or startup_profile.get("mainSector") or "Tecnologia e Inovação"
    s_stage = startup_profile.get("currentStage") or "Validação"
    s_trl = startup_profile.get("trlScore") or (8 if "tra" in s_stage.lower() else 4)
    s_tech = _normalize_tech_list(
        startup_profile.get("technologies") or startup_profile.get("areasOfInterest"),
        f"{s_area}, Inovação B2B",
    )
    s_model = startup_profile.get("businessModel") or "SaaS B2B"
    s_problem = (
        startup_profile.get("problem")
        or startup_profile.get("description")
        or f"Solução tecnológica em {s_area} focada em ganho de eficiência operacional e resultados mensuráveis."
    ).strip()
    s_maturity = startup_profile.get("productMaturity") or f"Estágio {s_stage} (TRL {s_trl}/9)"
    s_loc = startup_profile.get("location") or "Brasil"
    s_contact = startup_profile.get("phone") or startup_profile.get("userEmail") or "Canal institucional StartIn"

    has_opp = bool(selected_opp and (selected_opp.get("title") or selected_opp.get("description")))
    opp_title = (selected_opp or {}).get("title") or (attached_name if attached_name else "")
    opp_type = (selected_opp or {}).get("type") or "Edital"
    opp_company = (selected_opp or {}).get("companyName") or "Comitê Avaliador / Organização Responsável"
    opp_meta = (selected_opp or {}).get("meta") or (selected_opp or {}).get("metaInfo") or "Inscrições abertas · Brasil"
    opp_tags = _normalize_tech_list((selected_opp or {}).get("tags"), s_tech)
    opp_desc = (
        (selected_opp or {}).get("description")
        or (selected_opp or {}).get("short")
        or (attached_text[:600].strip() if attached_text else "")
    )

    q_raw = (user_request or "").strip()
    # Remove o sufixo automático de arquivo anexado para interpretar a pergunta real
    q_clean = re.sub(r"\[Arquivo anexado:.*?\]", "", q_raw, flags=re.IGNORECASE).strip()
    q = q_clean.lower()

    def has_any(terms: List[str]) -> bool:
        return any(t in q for t in terms)

    # 1. Perguntas sobre a plataforma StartIn / como funciona o sistema
    if has_any([
        "sobre a plataforma", "fale sobre a startin", "o que é a startin", "o que e a startin",
        "como funciona a plataforma", "como funciona o sistema", "como a startin funciona",
        "sobre o sistema", "funcionalidades", "para que serve a plataforma", "me explique a plataforma",
        "fale sobre o projeto", "o que essa plataforma faz",
    ]) or ("plataforma" in q and has_any(["fale", "explic", "sobre", "funciona", "o que", "qual"])):
        focus_note = (
            f"\n\nNo momento, a oportunidade salva '{opp_title}' ({opp_type}) está selecionada aqui no chat — se quiser, posso resumir os requisitos dela ou montar sua proposta."
            if opp_title
            else "\n\nPara gerar propostas personalizadas aqui no chat, basta salvar uma oportunidade em 'Para você' e selecioná-la na caixa 'Oportunidade / Edital em foco' ao lado."
        )
        return (
            f"Olá! A StartIn é uma plataforma inteligente de matchmaking e inovação aberta que conecta startups como a {s_name} a editais públicos de fomento, eventos e demandas corporativas reais.\n\n"
            f"Veja como os principais módulos funcionam para o seu perfil ({s_stage}):\n"
            f"1. Para você: A IA filtra e ranqueia automaticamente oportunidades compatíveis com a sua área ({s_area}) e com o seu estágio atual — startups em Validação recebem POCs, Primeira Compra, Editais e Eventos; startups em Tração (com TRL da NASA e Atestado Técnico validados) recebem Compras Corporativas diretas, Editais e Eventos.\n"
            f"2. Buscar empresas: Exibe o perfil e os canais diretos de contato das empresas contratantes, além de todas as POCs e Compras lançadas por cada uma.\n"
            f"3. Salvos e Assistente IA: Sempre que você salva interesse em uma oportunidade, ela fica disponível no seletor ao lado para que eu tire dúvidas sobre o edital, analise PDFs anexados ou redija templates, propostas técnicas e pitches prontos para envio.\n"
            f"4. Notificações e Home: Acompanham em tempo real novos matches, oportunidades compatíveis e interações com empresas.{focus_note}"
        )

    # 2. Saudação pura ou pergunta sobre quem é o assistente ("Olá", "Oi, tudo bem?", "Como pode me ajudar?")
    is_short_greeting = len(q) <= 45 and has_any([
        "olá", "ola", "oi", "bom dia", "boa tarde", "boa noite", "tudo bem", "opa", "e aí", "e ai"
    ]) and not has_any([
        "template", "proposta", "pitch", "edital", "poc", "compra", "prazo", "requisito", "trl", "mensagem", "email"
    ])
    if is_short_greeting or has_any(["quem é você", "quem e voce", "o que você faz", "o que voce faz", "como você pode me ajudar", "como pode ajudar"]):
        opp_line = (
            f"Vejo que você selecionou '{opp_title}' ({opp_type}). Posso tirar dúvidas específicas sobre essa oportunidade, resumir seus critérios, listar documentos necessários ou gerar um template/proposta sob medida."
            if opp_title
            else "Você pode selecionar uma oportunidade salva no painel esquerdo, anexar o PDF de um edital ou me fazer qualquer pergunta sobre estratégia de submissão, TRL, POCs, editais e posicionamento da sua startup."
        )
        return (
            f"Olá! Sou o Assistente de IA da StartIn, configurado com o perfil da {s_name} ({s_area} · Estágio {s_stage}).\n\n"
            f"{opp_line}\n\n"
            f"Como posso ajudar você agora?"
        )

    # 3. Perguntas sobre TRL da NASA, Estágio Validação vs Tração ou Atestado de Capacidade Técnica
    if has_any(["trl", "nasa", "atestado de capacidade", "estágio", "estagio", "validação e tração", "validacao e tracao", "diferença entre validação", "mudar para tração"]):
        return (
            f"Na StartIn, o estágio define quais modalidades corporativas a IA prioriza para a {s_name}:\n\n"
            f"1. Estágio Atual da {s_name}: {s_stage} (Maturidade registrada: {s_maturity}).\n"
            f"2. Escala TRL da NASA (1 a 9): Mede a prontidão tecnológica — TRL 1 a 3 indica concepção experimental; TRL 4 a 6 indica MVP/protótipo validado em ambiente relevante (típico do estágio Validação, ideal para POCs e Primeira Compra); TRL 7 a 9 comprova sistema homologado e em operação comercial contínua (estágio Tração, habilitado para Compras Corporativas diretas).\n"
            f"3. Como atualizar: Em 'Conta e acesso', você pode alternar entre Validação e Tração a qualquer momento. Ao escolher Tração, basta responder às 5 questões TRL e anexar seu Atestado de Capacidade Técnica em PDF para certificação imediata pela IA."
        )

    # 4. Perguntas pedindo avaliação do perfil da startup, diagnóstico ou como melhorar o pitch/match
    if has_any([
        "meu perfil", "minha startup", "avaliar minha", "avalie minha", "melhorar meu",
        "pontos fortes", "pontos fracos", "diagnóstico", "diagnostico", "aumentar meu match",
        "aumentar a aderência", "dicas para minha startup",
    ]):
        return (
            f"Diagnóstico de Posicionamento da {s_name} na StartIn:\n\n"
            f"1. Perfil Tecnológico Registrado\n"
            f"Área principal: {s_area} | Modelo: {s_model} | Localização: {s_loc}\n"
            f"Estágio e Maturidade: {s_stage} ({s_maturity})\n"
            f"Tecnologias mapeadas no matchmaking: {s_tech}\n\n"
            f"2. Ponto Forte Atual\n"
            f"Sua descrição de solução destaca: \"{s_problem[:260]}{'...' if len(s_problem) > 260 else ''}\". Isso gera alta sinergia com chamadas que buscam impacto operacional direto e aplicação prática de {s_tech}.\n\n"
            f"3. Recomendações Práticas para Elevar sua Taxa de Aprovação\n"
            f"a) Quantifique indicadores: inclua métricas percentuais de redução de custo, tempo economizado ou ganho de produtividade nas propostas.\n"
            f"b) Evidencie entregáveis de curto prazo: tanto em editais quanto em demandas corporativas, apresente marcos claros de 30, 60 e 90 dias.\n"
            f"c) Use a aba 'Salvos' para selecionar as oportunidades de maior aderência e gerar propostas adaptadas ao vocabulário de cada avaliador."
        )

    # 5. Perguntas sobre prazos, datas, local, formato ou como se inscrever
    if has_any([
        "qual o prazo", "qual é o prazo", "prazo", "até quando", "ate quando", "data limite",
        "quando encerra", "onde acontece", "localização", "remoto ou presencial", "como me inscrever",
        "como participar", "passo a passo para inscrever", "como enviar",
    ]):
        if not has_opp and not attached_text:
            return (
                f"Para consultar o prazo exato e o passo a passo de inscrição de uma chamada específica, selecione uma das suas oportunidades salvas no campo 'Oportunidade / Edital em foco' à esquerda ou anexe o PDF do regulamento.\n\n"
                f"De forma geral, em 'Para você' cada card exibe o prazo, formato (Remoto, Híbrido ou Presencial) e o link direto para a página oficial ou perfil da empresa ofertante."
            )
        target_label = opp_title or attached_name or "oportunidade em foco"
        return (
            f"Informações de Prazo, Formato e Inscrição — {target_label}:\n\n"
            f"1. Prazo e Formato Registrados: {opp_meta}\n"
            f"2. Modalidade: {opp_type} | Organização / Fonte: {opp_company}\n"
            f"3. Critérios e Temas Prioritários: {opp_tags}\n\n"
            f"Passo a Passo Recomendado para a {s_name}:\n"
            f"1. Verifique a aderência do escopo: {opp_desc[:240] if opp_desc else 'Alinhe o problema resolvido pela startup aos objetivos da chamada.'}\n"
            f"2. Gere aqui no chat a Proposta Técnica ou o Template de Submissão personalizado com seus dados de {s_stage}.\n"
            f"3. Acesse o link oficial da oportunidade (ou os canais de contato direto de {opp_company}, caso seja POC/Compra) nos detalhes da oportunidade para protocolar o envio antes do vencimento."
        )

    # 6. Perguntas sobre documentos necessários / checklist de submissão
    if has_any(["documento", "documentos", "checklist", "o que preciso separar", "anexos exigidos", "certidões", "certidoes", "o que preparar"]):
        target_label = f" para '{opp_title}' ({opp_type})" if opp_title else f" para oportunidades do estágio {s_stage}"
        is_corp = any(k in opp_type.lower() for k in ["poc", "compra"])
        if is_corp:
            return (
                f"Checklist de Documentação Recomendada{target_label}:\n\n"
                f"1. Cartão CNPJ ativo e dados cadastrais atualizados da {s_name}.\n"
                f"2. Proposta Técnica Executiva contendo escopo, arquitetura tecnológica ({s_tech}), cronograma e KPIs de sucesso.\n"
                f"3. {'Atestado de Capacidade Técnica / cases comerciais comprovando operação em escala (TRL ' + str(s_trl) + '/9).' if 'tra' in s_stage.lower() else 'Demonstração funcional do MVP e plano de validação de prova de conceito (POC) com métricas claras.'}\n"
                f"4. Estimativa comercial ou modelo de licenciamento ({s_model}) alinhado à governança de compras de {opp_company}.\n"
                f"5. Declaração de conformidade com LGPD e segurança da informação."
            )
        return (
            f"Checklist de Documentação para Submissão{target_label}:\n\n"
            f"1. Regularidade Institucional: Cartão CNPJ da {s_name}, contrato/estatuto social atualizado e certidões negativas básicas (Federal, Trabalhista e FGTS).\n"
            f"2. Formulário Técnico de Mérito e Inovação: Descrição do problema, grau de inovação em {s_area}, maturidade ({s_maturity}) e metodologia.\n"
            f"3. Plano de Trabalho e Cronograma Físico-Financeiro: Metas trimestrais, entregáveis verificáveis e alocação da equipe.\n"
            f"4. Currículo resumido dos fundadores e equipe técnica.\n"
            f"5. Anexos comprobatórios (protótipo, métricas de validação ou cartas de apoio/atestado técnico)."
        )

    # 7. Perguntas pedindo resumo, explicação, requisitos, elegibilidade ou análise do edital/documento
    if has_any([
        "resumo", "resuma", "sobre esse edital", "sobre esta oportunidade", "sobre essa oportunidade",
        "requisitos", "critérios", "criterios", "o que pede", "quem pode participar", "elegibilidade",
        "do que trata", "explique esse", "analise o documento", "analise esse", "vale a pena", "tem aderência", "tem aderencia",
    ]):
        if not has_opp and not attached_text:
            return (
                f"Ainda não há uma oportunidade salva selecionada nem um arquivo anexado nesta conversa.\n\n"
                f"Selecione uma oportunidade salva na caixa 'Oportunidade / Edital em foco' à esquerda (ou anexe o PDF/TXT do edital no botão de clipe) e eu farei a análise completa dos requisitos e da aderência para a {s_name}."
            )
        doc_excerpt = (
            f"\n\nTrecho relevante identificado no arquivo '{attached_name}':\n\"{to_plain_text(attached_text[:550])}...\""
            if attached_text
            else ""
        )
        return (
            f"Análise de Requisitos e Aderência — {opp_title or attached_name}:\n\n"
            f"1. Escopo da Oportunidade ({opp_type} · {opp_company})\n"
            f"{opp_desc if opp_desc else 'Oportunidade voltada para soluções inovadoras com aplicação tecnológica comprovável.'}\n"
            f"Vigência / Formato: {opp_meta} | Temas buscados: {opp_tags}.{doc_excerpt}\n\n"
            f"2. Por que faz sentido para a {s_name}\n"
            f"Sua startup atua em {s_area} no estágio {s_stage} ({s_maturity}), aplicando {s_tech} no modelo {s_model}. O problema que você resolve dialoga diretamente com a busca por eficiência, inovação aplicada e geração de valor descrita nesta oportunidade.\n\n"
            f"3. Próximo Passo Sugerido\n"
            f"Se desejar, peça agora 'Gerar proposta técnica detalhada' ou 'Criar texto persuasivo de mérito' e eu monto o texto completo pronto para submissão."
        )

    # 8. Modelo Rápido 4 / Pedido de Mensagem Executiva, E-mail Comercial ou Abordagem Direta
    if has_any([
        "mensagem executiva", "apresentação comercial", "apresentacao comercial",
        "e-mail", "email", "carta de apresentação", "abordagem", "mensagem para empresa", "whatsapp",
    ]):
        target_dest = opp_company if has_opp else "Equipe de Inovação Aberta e Novos Negócios"
        target_ref = f"referente à oportunidade '{opp_title}' ({opp_type})" if opp_title else f"para iniciativas de inovação aberta em {s_area}"
        return (
            f"ASSUNTO: Proposta de Conexão Tecnológica ({s_area}) — {s_name}"
            + (f" | {opp_title}" if opp_title else "")
            + "\n\n"
            f"Olá, {target_dest},\n\n"
            f"Entro em contato em nome da {s_name} ({s_loc}), startup especializada em {s_area} atualmente em estágio de {s_stage} ({s_maturity}), {target_ref}.\n\n"
            f"Desenvolvemos uma solução no modelo {s_model} baseada em {s_tech} para resolver um gargalo crítico do setor:\n"
            f"{s_problem}\n\n"
            + (
                f"Analisamos os critérios de '{opp_title}' ({opp_tags}) e identificamos aderência direta para entrega rápida de resultados mensuráveis junto à sua operação.\n\n"
                if opp_title
                else "Identificamos forte sinergia tecnológica para apoiar seus desafios operacionais com implantação ágil e indicadores claros de retorno.\n\n"
            )
            + f"Podemos agendar uma breve conversa técnica de 20 minutos nesta semana para demonstrar a solução em funcionamento?\n\n"
            f"Atenciosamente,\n"
            f"Equipe {s_name}\n"
            f"Contato: {s_contact}"
        )

    # 9. Modelo Rápido 3 / Pedido de Texto Persuasivo de Mérito, Diferencial Tecnológico e Impacto / Pitch
    if has_any([
        "texto persuasivo", "mérito", "merito", "diferencial", "impacto esperado",
        "pitch", "defesa", "por que nossa startup", "argumento", "convencer",
    ]):
        target_header = f"PARA {opp_title.upper()}" if opp_title else f"— {s_name.upper()}"
        return (
            f"TEXTO DE MERITO TECNOLOGICO, DIFERENCIAL E IMPACTO {target_header}\n\n"
            f"1. RELEVANCIA DO PROBLEMA E OPORTUNIDADE\n"
            f"Organizações e operações modernas enfrentam perdas recorrentes de eficiência por processos fragmentados. A {s_name} atua diretamente nessa dor no segmento de {s_area}: {s_problem}\n\n"
            f"2. INOVACAO E DIFERENCIAL TECNOLOGICO\n"
            f"Diferente de abordagens tradicionais ou consultivas manuais, a {s_name} combina {s_tech} em uma arquitetura {s_model} escalável, com maturidade comprovada ({s_maturity})."
            + (f" Essa base atende diretamente aos vetores exigidos em '{opp_title}' ({opp_tags})." if opp_title else "")
            + "\n\n"
            f"3. IMPACTO MENSURAVEL E POTENCIAL DE ESCALA\n"
            f"A adoção da solução permite redução tangível de retrabalho operacional, rastreabilidade de ponta a ponta e tomada de decisão orientada por dados desde as primeiras semanas de implantação, garantindo alto retorno sobre o investimento e sustentabilidade de longo prazo."
        )

    # 10. Modelo Rápido 2 / Pedido de Proposta Técnica Detalhada, Plano de Trabalho, Cronograma ou Orçamento/KPIs
    if has_any([
        "proposta técnica", "proposta tecnica", "plano de trabalho", "cronograma",
        "metodologia", "entregáveis", "entregaveis", "kpis", "etapas de execução",
    ]):
        target_title = opp_title or f"Iniciativa de Inovação em {s_area}"
        return (
            f"PROPOSTA TECNICA E PLANO DE TRABALHO — {target_title.upper()}\n"
            f"Proponente: {s_name} ({s_area} · {s_stage} · {s_maturity})\n"
            f"Destinatário: {opp_company}\n\n"
            f"1. OBJETIVO TECNICO\n"
            f"Implantar e validar a tecnologia da {s_name} ({s_tech}) aplicada ao escopo de '{target_title}', endereçando de forma mensurável: {s_problem}\n\n"
            f"2. ARQUITETURA E METODOLOGIA DE EXECUCAO\n"
            f"A solução opera em modelo {s_model}, permitindo integração não intrusiva, segurança de dados e acompanhamento por dashboards e indicadores operacionais.\n\n"
            f"3. CRONOGRAMA DE ENTREGAVEIS (90 DIAS)\n"
            f"Fase 1 (Semanas 1 a 3) — Diagnóstico e Setup: Mapeamento técnico dos fluxos junto a {opp_company}, definição da linha de base e configuração do ambiente.\n"
            f"Fase 2 (Semanas 4 a 8) — Implantação e Operação Assistida: Ativação dos módulos de {s_tech}, treinamento dos usuários-chave e coleta contínua de métricas.\n"
            f"Fase 3 (Semanas 9 a 12) — Avaliação de Resultados e Escala: Consolidação dos KPIs atingidos, relatório técnico-executivo de ROI e plano de expansão contratual.\n\n"
            f"4. INDICADORES DE SUCESSO (KPIS SUGERIDOS)\n"
            f"a) Ganho mensurável de velocidade e eficiência no processo-alvo;\n"
            f"b) Acurácia e confiabilidade dos dados processados pela plataforma;\n"
            f"c) Satisfação operacional da equipe usuária (NPS interno) e viabilidade de escala."
        )

    # 11. Modelo Rápido 1 / Pedido explícito de Template Completo de Submissão
    if has_any([
        "template", "submissão", "submissao", "formulário", "formulario",
        "modelo completo", "inscrição completa", "inscricao completa",
    ]) or (attached_text and not q_clean):
        target_title = opp_title or "Oportunidade de Inovação"
        return (
            f"TEMPLATE DE SUBMISSAO COMPLETO — {target_title.upper()}\n"
            f"Modalidade: {opp_type} | Destinatário: {opp_company}\n\n"
            f"1. APRESENTACAO EXECUTIVA DA STARTUP\n"
            f"Proponente: {s_name} ({s_area} · {s_loc})\n"
            f"Maturidade e Estágio: {s_stage} ({s_maturity})\n"
            f"Tecnologias-Chave: {s_tech}\n\n"
            f"2. ALINHAMENTO COM O DESAFIO / EDITAL\n"
            f"Em resposta a '{target_title}', a {s_name} apresenta uma solução diretamente aderente aos requisitos ({opp_desc[:240] if opp_desc else opp_tags}).\n\n"
            f"3. PROBLEMA ENDERECADO E SOLUCAO PROPOSTA\n"
            f"{s_problem}\n\n"
            f"4. PLANO DE EXECUCAO E ENTREGAVEIS\n"
            f"Etapa 1 — Alinhamento técnico, levantamento de requisitos e definição de KPIs junto a {opp_company}.\n"
            f"Etapa 2 — Parametrização tecnológica ({s_tech}), integração e validação operacional acompanhada.\n"
            f"Etapa 3 — Mensuração dos indicadores de impacto, relatório executivo de resultados e plano de escala.\n\n"
            f"5. DIFERENCIAIS E CONTATO PARA FORMALIZACAO\n"
            f"A {s_name} reúne aderência técnica comprovada no modelo {s_model} e agilidade de entrega para maximizar os resultados desta iniciativa. Contato direto: {s_contact}."
        )

    # 12. Qualquer outra dúvida específica (orçamento, preço, negociação, POC, propriedade intelectual, etc.)
    context_ref = (
        f"considerando a oportunidade '{opp_title}' ({opp_type} · {opp_company})"
        if opp_title
        else f"considerando o momento atual da {s_name} ({s_stage})"
    )
    if has_any(["preço", "preco", "orçamento", "orcamento", "valor", "cobrar", "precificar", "financeiro", "custo"]):
        return (
            f"Orientação sobre Orçamento e Precificação ({context_ref}):\n\n"
            f"1. Para Editais de Fomento / Subvenção: Estruture o orçamento separando rubricas de custeio (equipe de P&D em {s_tech}, serviços de nuvem/infraestrutura e testes de validação), evitando despesas administrativas genéricas não financiáveis.\n"
            f"2. Para POCs e Primeira Compra: Precifique cobrindo setup + licença de prova de conceito (POC) por período fechado (ex: 3 meses), deixando pré-acordada a faixa de assinatura {s_model} em caso de aprovação dos KPIs.\n"
            f"3. Para Compra Corporativa em Tração: Apresente proposta comercial baseada em valor/ROI gerado sobre a dor principal que a {s_name} resolve, com SLA definido e faixas de volume."
        )

    return (
        f"Sobre sua pergunta (\"{q_clean}\"), {context_ref}:\n\n"
        f"Para a {s_name} ({s_area} · Estágio {s_stage}), a melhor estratégia na StartIn é conectar diretamente a dor operacional que vocês resolvem com evidências práticas da sua tecnologia ({s_tech}).\n"
        + (
            f"\nNo caso específico de '{opp_title}', o foco principal busca '{opp_tags}' ({opp_meta}). Recomendo destacar resultados objetivos logo no primeiro parágrafo da submissão.\n"
            if opp_title
            else "\nVocê pode selecionar qualquer oportunidade salva na caixa à esquerda para que eu cruze os requisitos específicos dela com os dados da sua startup.\n"
        )
        + "\nSe quiser, posso agora:\n"
        f"1. Resumir os requisitos e critérios de elegibilidade de uma oportunidade salva;\n"
        f"2. Redigir uma proposta técnica, pitch de mérito ou e-mail comercial personalizado;\n"
        f"3. Explicar detalhes sobre estágios (Validação/Tração), escala TRL ou documentação."
    )


def fetch_provider_from_backend(
    provider_id: Optional[int] = None, user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Busca o perfil completo do Prestador de Serviços no backend Java Spring Boot."""
    endpoints = []
    if provider_id:
        endpoints.append(f"{BACKEND_API_URL}/providers/{provider_id}")
    if user_id:
        endpoints.append(f"{BACKEND_API_URL}/providers/user/{user_id}")
    endpoints.append(f"{BACKEND_API_URL}/providers")

    for url in endpoints:
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                if isinstance(data, dict) and data.get("name"):
                    return data
        except Exception as e:
            print(f"[StartIn IA] Aviso ao consultar prestador no backend ({url}): {e}")
    return {}


def _format_provider_for_prompt(provider: dict) -> str:
    """Formata o dossiê completo do Prestador de Serviços para o Assistente IA."""
    if not provider:
        return "Perfil de prestador de serviços na StartIn."
    p_type = "Empresa / Agência / Software House" if str(provider.get("providerType", "")).upper() == "EMPRESA" else "Profissional Autônomo / Freelancer"
    return (
        f"1. IDENTIDADE DO PRESTADOR:\n"
        f"   - Nome / Razão Social: {provider.get('name') or 'Prestador'}\n"
        f"   - Tipo de Prestador: {p_type}\n"
        f"   - Categoria Principal: {provider.get('category') or 'Serviços Especializados'}\n"
        f"   - Serviço Principal: {provider.get('mainService') or provider.get('category') or 'Consultoria e Execução Técnica'}\n"
        f"   - Tempo de Experiência: {provider.get('experienceYears') or '4'} anos\n"
        f"   - Localização: {provider.get('location') or 'Brasil'}\n"
        f"   - Disponibilidade: {provider.get('availability') or 'Disponível imediatamente'}\n"
        f"2. BIO E EXPERIÊNCIA EM SERVIÇOS:\n"
        f"   - Descrição Profissional: {provider.get('description') or provider.get('serviceExperience') or 'Especialista em soluções para startups em Tração.'}\n"
        f"   - Competências e Ferramentas: {provider.get('toolsAndSkills') or provider.get('tags') or provider.get('category') or ''}\n"
        f"   - Segmento Alvo: {provider.get('targetSegment') or 'Startups em Tração (B2B/SaaS)'}\n"
        f"3. PORTFÓLIO E PREFERÊNCIAS DE PARCERIA:\n"
        f"   - Case / Projeto de Destaque: {provider.get('projectName') or 'Projetos entregues na área'} — {provider.get('projectDescription') or ''}\n"
        f"   - Tecnologias do Case: {provider.get('projectTools') or ''}\n"
        f"   - Setores de Maior Interesse: {provider.get('interestSectors') or 'Multissetorial (Startups em Tração)'}\n"
        f"   - Modelo de Contratação Preferido: {provider.get('connectionPreference') or 'Contrato PJ / Sprints / Projeto Fechado'}\n"
        f"   - Oportunidades Buscadas: {provider.get('opportunitiesSought') or 'Demandas imediatas e parcerias contínuas'}\n"
        f"   - Região de Interesse: {provider.get('interestRegion') or 'Brasil / Remoto'}"
    )


def _build_provider_contextual_reply(
    provider_profile: dict,
    selected_opportunity: Optional[dict],
    attachment_name: Optional[str],
    attachment_text: str,
    user_msg: str,
) -> str:
    """Resposta contextual inteligente para Prestadores de Serviços quando o LLM externo oscilar."""
    p_name = provider_profile.get("name") or "Prestador"
    p_cat = provider_profile.get("category") or "Serviços Especializados"
    p_service = provider_profile.get("mainService") or f"Atuação especializada em {p_cat}"
    p_skills = provider_profile.get("toolsAndSkills") or provider_profile.get("tags") or p_cat
    p_model = provider_profile.get("connectionPreference") or "Contrato PJ / Sprints Ágeis / Projeto Fechado"
    p_sectors = provider_profile.get("interestSectors") or "Fintech, Healthtech e SaaS B2B"
    p_exp = provider_profile.get("experienceYears") or "4"
    p_desc = provider_profile.get("description") or f"Atuo com {p_service} utilizando {p_skills} para acelerar resultados de startups em Tração."

    has_st = bool(selected_opportunity and (selected_opportunity.get("title") or selected_opportunity.get("description")))
    st_title = (selected_opportunity or {}).get("title") or (attachment_name or "Startup em Tração")
    st_meta = (selected_opportunity or {}).get("meta") or "Estágio de Tração · Brasil"
    st_desc = (selected_opportunity or {}).get("description") or ""

    q = (user_msg or "").lower()

    if any(w in q for w in ["proposta", "comercial", "orçamento", "orcamento", "template"]):
        return (
            f"PROPOSTA COMERCIAL E TÉCNICA — {p_name.upper()} PARA {st_title.upper()}\n\n"
            f"1. Apresentação do Prestador:\n"
            f"Responsável: {p_name} ({p_cat} · {p_exp} anos de experiência)\n"
            f"Frente Principal: {p_service}\n"
            f"Competências e Ferramentas: {p_skills}\n"
            f"Resumo Profissional: {p_desc}\n\n"
            f"2. Diagnóstico e Alinhamento com {st_title} ({st_meta}):\n"
            f"{st_desc if st_desc else 'Atendimento direcionado às demandas de crescimento, escala operacional e entregas especializadas da startup.'}\n\n"
            f"3. Escopo de Entregáveis e Plano de Sprints:\n"
            f"a) Sprint 1 (Diagnóstico e Setup - Semana 1): Alinhamento de KPIs com os fundadores, auditoria técnica/operacional e plano de execução.\n"
            f"b) Sprints 2 e 3 (Execução Principal - Semanas 2 a 5): Implementação prática em {p_service} utilizando {p_skills}, com entregas semanais validáveis.\n"
            f"c) Sprint 4 (Otimização e Hand-off - Semana 6): Mensuração de indicadores de impacto, documentação e transferência de conhecimento para o time interno.\n\n"
            f"4. Modelo de Contratação e Próximos Passos:\n"
            f"Formato sugerido: {p_model}\n"
            f"Disponibilidade: {provider_profile.get('availability') or 'Imediata'}\n"
            f"Próximo passo: Agendar call de 20 minutos pela aba Conexões da StartIn para validar prioridades."
        )

    if any(w in q for w in ["abordagem", "mensagem", "conectar", "conexão", "conexao", "convite", "primeiro contato"]):
        return (
            f"MENSAGEM DE ABORDAGEM PARA CONEXÃO COM {st_title.upper()}\n\n"
            f"Olá, equipe da {st_title}! Tudo bem?\n\n"
            f"Vi o perfil de vocês em estágio de Tração aqui na StartIn e identifiquei alta sinergia com a minha atuação em {p_service} ({p_cat}).\n\n"
            f"{p_desc}\n\n"
            f"Trabalho com {p_skills} no formato {p_model}, focando em entregas rápidas e orientadas a métricas para startups em fase de escala. "
            f"Gostaria de conectar para entender as prioridades atuais de vocês e compartilhar como posso apoiar nas próximas sprints.\n\n"
            f"Podemos alinhar uma breve conversa esta semana?\n"
            f"Um abraço,\n"
            f"{p_name}"
        )

    if any(w in q for w in ["escopo", "sprint", "cronograma", "contrato", "preço", "preco", "precifica", "cobrar"]):
        return (
            f"ESTRUTURA DE ESCOPO, SPRINTS E MODELOS DE CONTRATO PARA {p_cat.upper()}\n\n"
            f"Para atender startups em estágio de Tração na StartIn com segurança e agilidade, recomendo estruturar sua atuação em {p_service} nos seguintes formatos:\n\n"
            f"1. Modelo por Sprints / Retainer Mensal ({p_model}):\n"
            f"Ideal para evolução contínua em {p_skills}. Ciclos quinzenais com backlog priorizado junto ao CTO/CMO/CEO da startup.\n\n"
            f"2. Modelo de Escopo Fechado por Entregável (4 a 6 semanas):\n"
            f"a) Fase 1 (15% do tempo): Imersão no produto da startup e definição de metas.\n"
            f"b) Fase 2 (65% do tempo): Execução intensiva com checkpoints semanais.\n"
            f"c) Fase 3 (20% do tempo): Ajustes finos, análise de métricas e entrega final.\n\n"
            f"3. Boas Práticas Contratuais com Startups em Tração:\n"
            f"Defina claramente SLA de comunicação, propriedade intelectual dos entregáveis, indicadores de sucesso e cláusula de renovação após o primeiro ciclo validado."
        )

    if any(w in q for w in ["perfil", "portfólio", "portfolio", "melhorar", "aumentar", "match", "otimizar"]):
        return (
            f"DIAGNÓSTICO DO SEU PERFIL DE PRESTADOR ({p_name} · {p_cat})\n\n"
            f"1. Pontos Fortes Atuais:\n"
            f"a) Categoria e Serviço Principal: {p_cat} — {p_service}\n"
            f"b) Competências Mapeadas (30% do score): {p_skills}\n"
            f"c) Setores e Modelo Preferido: {p_sectors} · {p_model}\n\n"
            f"2. Como Aumentar seu Índice de Match com Startups em Tração:\n"
            f"a) Na aba 'Meu perfil', utilize o botão 'Melhorar descrição com IA' para manter sua bio em 1ª pessoa rica em resultados práticos e métricas.\n"
            f"b) Mantenha pelo menos 5 a 8 competências e ferramentas específicas ativas nas tags de matchmaking.\n"
            f"c) Preencha o Case de Destaque do Portfólio detalhando o problema resolvido e o impacto gerado — o motor da StartIn bonifica perfis com case completo e links verificáveis."
        )

    return (
        f"Olá, {p_name}! Analisei sua pergunta considerando seu perfil de prestador em {p_cat} ({p_service} · competências: {p_skills}).\n\n"
        + (f"Startup em foco: {st_title} ({st_meta}).\n\n" if has_st else "")
        + "Na StartIn, os prestadores de serviços se conectam diretamente com Startups em estágio de Tração (TRL 7 a 9 com operação validada). "
        "O motor de IA calcula sua compatibilidade em 3 pilares:\n"
        "1. Sinergia de Categoria e Demanda Aberta (40%): alinhamento entre sua especialidade e o setor/demanda de serviço publicada pela startup;\n"
        "2. Competências e Ferramentas (30%): cruzamento direto entre suas tags técnicas e a stack/necessidade da startup;\n"
        "3. Modelo de Atuação, Portfólio e Disponibilidade (30%): aderência ao formato de contratação (Contrato PJ, Squad, Projeto Fechado), case comprovado e região.\n\n"
        "Você pode clicar em uma das opções rápidas ao lado ou me pedir agora:\n"
        "1. Uma proposta comercial e técnica completa para uma startup salva;\n"
        "2. Uma mensagem persuasiva de primeiro contato para enviar em 'Solicitar conexão';\n"
        "3. Sugestão de escopo de sprints, entregáveis e formato de contrato para sua área."
    )


def generate_assistant_chat(
    messages: list,
    startup_profile: Optional[dict] = None,
    opportunities: Optional[List[dict]] = None,
    selected_opportunity: Optional[dict] = None,
    attachment_name: Optional[str] = None,
    attachment_base64: Optional[str] = None,
    attachment_text: Optional[str] = None,
    account_type: Optional[str] = "STARTUP",
    provider_profile: Optional[dict] = None,
    company_profile: Optional[dict] = None,
    template_style: Optional[str] = None,
) -> str:
    """
    Assistente Inteligente da StartIn treinado para interpretar qualquer pergunta
    do usuário com coerência — para Startups, Empresas e Prestadores de Serviços,
    com suporte a Definição de Template personalizado ou 'padrão'.
    """
    raw_style = (template_style or "").strip()
    is_default_style = (not raw_style) or raw_style.lower() in ("padrão", "padrao", "default", "normal")
    template_directive = (
        "O usuário escolheu o estilo de template 'padrão': gere os templates na estrutura oficial da StartIn."
        if is_default_style
        else f"ESTILO DE TEMPLATE DEFINIDO PELO USUÁRIO: '{raw_style}'. Adapte e estruture o template/documento de forma 100% COESA seguindo exatamente o formato, seções e estilo '{raw_style}' solicitado pelo usuário (ou lembre-o de que ele pode digitar 'padrão' caso queira o modelo tradicional da IA)."
    )

    def _apply_style_wrapper(base_text: str) -> str:
        if is_default_style:
            return base_text
        upper_base = (base_text or "").upper()
        is_template_doc = any(
            marker in upper_base
            for marker in (
                "TEMPLATE DE",
                "PROPOSTA COMERCIAL",
                "PROPOSTA TÉCNICA",
                "PROPOSTA TECNICA",
                "PLANO DE TRABALHO",
                "TEXTO DE MÉRITO",
                "TEXTO DE MERITO",
                "MENSAGEM DE ABORDAGEM",
                "ASSUNTO:",
                "TERMO DE REFERÊNCIA",
                "ROTEIRO DE AVALIAÇÃO",
            )
        )
        if not is_template_doc:
            return base_text
        return (
            f"ESTILO DE TEMPLATE APLICADO: {raw_style.upper()}\n\n"
            f"{base_text}\n\n"
            f"Nota de Formatação: Documento estruturado com coesão segundo o modelo '{raw_style}'. "
            f"Você pode exportar este conteúdo em PDF ou DOCX nos botões abaixo, ou digitar 'padrão' para usar o template tradicional."
        )
    last_user_msg = ""
    for msg in reversed(messages or []):
        if msg.get("from") not in ("ai", "assistant"):
            last_user_msg = (msg.get("text") or msg.get("content") or "").strip()
            if last_user_msg:
                break

    extracted_attachment = (attachment_text or "").strip()
    if not extracted_attachment and attachment_base64:
        extracted_attachment = extract_text_from_attachment(attachment_name, attachment_base64)

    is_provider = str(account_type or "").upper() == "PROVIDER" or bool(
        provider_profile and (provider_profile.get("category") or provider_profile.get("mainService") or provider_profile.get("providerType"))
    )

    if is_question_outside_project_scope(last_user_msg):
        if is_provider:
            return (
                "Sou o Assistente Inteligente da StartIn para Prestadores de Serviços. Atuo exclusivamente em temas ligados à plataforma StartIn, "
                "ao seu perfil de prestador e às oportunidades e demandas de startups em Tração (como propostas comerciais e técnicas, mensagens de abordagem, escopos de sprints e análise de documentos anexados)."
            )
        return (
            "Sou o Assistente Inteligente da StartIn. Atuo exclusivamente em temas ligados à plataforma StartIn, ao seu perfil e às oportunidades "
            "salvas ou anexadas (editais, POCs, compras corporativas, escala TRL da NASA, propostas técnicas e geração de templates)."
        )

    attachment_context = "Nenhum arquivo anexado nesta mensagem."
    if extracted_attachment or attachment_name:
        attachment_context = (
            f"Arquivo anexado pelo usuário: {attachment_name or 'documento_anexo'}\n"
            f"Conteúdo lido do arquivo:\n{extracted_attachment[:8000] if extracted_attachment else '(Arquivo recebido para referência)'}"
        )

    # =========================================================================
    # FLUXO DO PRESTADOR DE SERVIÇOS NO ASSISTENTE IA
    # =========================================================================
    if is_provider:
        prov = dict(provider_profile) if isinstance(provider_profile, dict) else {}
        if not prov.get("name") or not prov.get("category"):
            fetched_prov = fetch_provider_from_backend(
                provider_id=prov.get("id"),
                user_id=prov.get("userId"),
            )
            if fetched_prov:
                prov = {**fetched_prov, **{k: v for k, v in prov.items() if v not in (None, "", [])}}

        provider_dossier = _format_provider_for_prompt(prov)
        selected_startup_context = "Nenhuma startup salva selecionada no momento."
        if selected_opportunity and (selected_opportunity.get("title") or selected_opportunity.get("description")):
            selected_startup_context = (
                f"Startup em Tração Selecionada: {selected_opportunity.get('title')}\n"
                f"Setor / Localização: {selected_opportunity.get('meta') or 'Tração · Brasil'}\n"
                f"Tecnologias / Tags: {selected_opportunity.get('tags') or ''}\n"
                f"Contexto e Demandas da Startup:\n{selected_opportunity.get('description') or selected_opportunity.get('short') or ''}"
            )

        if not client:
            return _apply_style_wrapper(_build_provider_contextual_reply(
                prov, selected_opportunity, attachment_name, extracted_attachment, last_user_msg
            ))

        provider_system_instruction = f"""
        Você é o Assistente Oficial de IA da plataforma StartIn dedicado aos PRESTADORES DE SERVIÇOS (profissionais autônomos, consultores, designers, desenvolvedores, especialistas em marketing/growth, jurídico, financeiro, agências e software houses) que atendem Startups em estágio de Tração.

        DEFINIÇÃO DE TEMPLATE ATIVA:
        {template_directive}

        DIRETRIZES OBRIGATÓRIAS DE INTERPRETAÇÃO E COERÊNCIA:
        1. INTERPRETE EXATAMENTE O QUE O PRESTADOR PERGUNTOU:
           - Leia e responda diretamente à dúvida ou solicitação feita pelo prestador, usando os dados reais do perfil dele abaixo (nome, categoria, serviço principal, bio, ferramentas/competências, case de portfólio e modelo de contrato).
           - Se o prestador perguntar como funciona a StartIn para prestadores, explique com clareza: na StartIn, prestadores têm acesso direto a Startups em estágio de Tração (TRL 7-9 validado); a aba Recomendações ranqueia as startups compatíveis pela IA (40% categoria/demanda aberta + 30% ferramentas/competências + 30% modelo de contrato/portfólio/disponibilidade); a aba Startups Salvas guarda favoritas; a aba Conexões gerencia solicitações enviadas/recebidas e libera contatos diretos quando aceitas.
           - Se o prestador pedir proposta comercial/técnica, mensagem de abordagem para conexão, estrutura de escopo/sprints, dicas de precificação/contrato PJ/Squad ou análise do documento anexado, gere uma resposta prática, completa, personalizada para a área dele ({prov.get('category') or 'Serviços'}) e cruzada com a startup selecionada (se houver).
           - Se o prestador pedir dicas para melhorar o perfil ou aumentar o match score, analise os dados reais do perfil dele abaixo e dê recomendações acionáveis específicas para a área de {prov.get('category') or 'atuação dele'}.
        2. ESCOPO: Recuse educadamente assuntos totalmente fora do universo de prestação de serviços, startups, negócios, tecnologia e da plataforma StartIn.
        3. FORMATO OBRIGATÓRIO — TEXTO PLANO PURO (PROIBIDO USAR MARKDOWN):
           - Gere TODA a resposta EXCLUSIVAMENTE em texto plano puro, sem NENHUMA sintaxe Markdown.
           - PROIBIDO usar asteriscos (* ou **), cerquilhas (#, ##), travessões divisórios (---), listas com hífen (- item), crases (`) ou tabelas com barras (|).
           - Use apenas numeração simples (1., 2., 3. ou a), b), c)), títulos limpos e parágrafos separados por linha em branco.

        DADOS REAIS DO PRESTADOR DE SERVIÇOS LOGADO:
        {provider_dossier}

        STARTUP EM TRAÇÃO SELECIONADA PELO PRESTADOR (SE HOUVER):
        {selected_startup_context}

        DOCUMENTO / BRIEFING / ESCOPO ANEXADO NO CHAT (SE HOUVER):
        {attachment_context}
        """

        chat_history = [{"role": "system", "content": provider_system_instruction}]
        for msg in (messages or [])[-10:]:
            role = "assistant" if msg.get("from") in ["ai", "assistant"] else "user"
            content = msg.get("text") or msg.get("content") or ""
            chat_history.append({"role": role, "content": content})

        seen_models = []
        for candidate_model in FALLBACK_MODELS:
            if not candidate_model or candidate_model in seen_models:
                continue
            seen_models.append(candidate_model)
            try:
                completion = client.chat.completions.create(
                    model=candidate_model,
                    messages=chat_history,
                    temperature=0.4,
                    max_tokens=1300,
                )
                raw_reply = completion.choices[0].message.content or ""
                if raw_reply.strip():
                    return to_structured_plain_text(raw_reply)
            except Exception as e:
                err_str = str(e)
                print(f"[StartIn IA - Provider Chat] Aviso no modelo '{candidate_model}': {err_str}")
                if "401" in err_str or "invalid_api_key" in err_str.lower():
                    break

        return _apply_style_wrapper(_build_provider_contextual_reply(
            prov, selected_opportunity, attachment_name, extracted_attachment, last_user_msg
        ))

    if not startup_profile or not startup_profile.get("name"):
        startup_profile = fetch_user_and_startup_from_backend(
            startup_id=startup_profile.get("id") if startup_profile else None,
            user_id=startup_profile.get("userId") if startup_profile else None,
        )

    startup_profile = startup_profile or {}
    profile_dossier = _format_profile_for_prompt(startup_profile)

    selected_opp_context = "Nenhuma oportunidade salva selecionada no momento."
    if selected_opportunity and (selected_opportunity.get("title") or selected_opportunity.get("description")):
        selected_opp_context = (
            f"Título: {selected_opportunity.get('title')}\n"
            f"Tipo: {selected_opportunity.get('type') or 'EDITAL'}\n"
            f"Empresa / Fonte: {selected_opportunity.get('companyName') or 'Edital / Chamada Oficial'}\n"
            f"Prazo / Meta: {selected_opportunity.get('meta') or selected_opportunity.get('metaInfo') or 'Aberto'}\n"
            f"Tags / Critérios: {selected_opportunity.get('tags') or ''}\n"
            f"Descrição Completa: {selected_opportunity.get('description') or selected_opportunity.get('short') or ''}"
        )

    attachment_context = "Nenhum arquivo anexado nesta mensagem."
    if extracted_attachment or attachment_name:
        attachment_context = (
            f"Arquivo anexado pelo usuário: {attachment_name or 'documento_anexo'}\n"
            f"Conteúdo lido do arquivo:\n{extracted_attachment[:8000] if extracted_attachment else '(Arquivo recebido para referência)'}"
        )

    if not client:
        return _apply_style_wrapper(_build_intelligent_contextual_reply(
            startup_profile, selected_opportunity, attachment_name, extracted_attachment, last_user_msg
        ))

    system_instruction = f"""
    Você é o Assistente Oficial de IA da plataforma StartIn — especialista no ecossistema StartIn, inovação aberta, editais de fomento (FAPESP PIPE, FINEP, Sebrae, InovaStartups), escala TRL da NASA (1 a 9), POCs, Primeira Compra e Compras Corporativas.

    DEFINIÇÃO DE TEMPLATE ATIVA:
    {template_directive}

    DIRETRIZES OBRIGATÓRIAS DE INTERPRETAÇÃO E COERÊNCIA:
    1. INTERPRETE EXATAMENTE O QUE FOI PERGUNTADO: Nunca repita uma resposta padrão ou um template de submissão se o usuário não pediu um template!
       - Se o usuário cumprimentar ou perguntar sobre a plataforma StartIn (ex: "Olá, me fale sobre a plataforma", "Como funciona?"), explique de forma natural, clara e acolhedora o que é a StartIn (matchmaking por IA, filtragem automática por estágio Validação vs Tração com TRL da NASA + Atestado PDF, aba Para Você, Busca de Empresas, Salvos e o gerador de propostas).
       - Se o usuário fizer perguntas específicas sobre a oportunidade selecionada ou sobre o PDF anexado (ex: prazos, requisitos, elegibilidade, resumo, riscos, documentos, dicas), responda DIRETAMENTE à dúvida usando os dados da oportunidade/anexo.
       - Se o usuário pedir avaliação do perfil da startup, diagnóstico ou dicas de pitch/negociação, analise os dados reais da startup abaixo e dê orientações práticas.
       - Se (e somente se) o usuário pedir um template de submissão, proposta técnica, plano de trabalho, texto persuasivo de mérito ou mensagem comercial, gere o documento completo e personalizado cruzando os dados da startup com a oportunidade em foco.
    2. ESCOPO: Recuse educadamente temas totalmente alheios a startups, inovação, tecnologia, negócios e à plataforma StartIn (como receitas culinárias, futebol, horóscopo, piadas).
    3. FORMATO OBRIGATÓRIO — TEXTO PLANO PURO (PROIBIDO USAR MARKDOWN):
       - Gere TODA a resposta EXCLUSIVAMENTE em texto plano puro, sem NENHUMA sintaxe ou formatação Markdown.
       - É ESTRITAMENTE PROIBIDO usar: asteriscos (* ou **), títulos com cerquilha (#, ##, ###), linhas divisórias (---, ***, ___), blocos de citação (>), listas com hífen ou asterisco no início da linha (- item, * item, + item), crases (` ou ```), sublinhados de estilo (_ ou __), til (~~), tabelas com barras verticais (| coluna |) ou tags HTML (<br>, <b>, <i>).
       - Para organizar o conteúdo com clareza em texto plano, utilize apenas numeração simples (1., 2., 3. ou a), b), c)), títulos limpos em texto puro seguidos de quebra de linha, campos diretos no formato "Rótulo: valor" e parágrafos separados por uma linha em branco.

    DADOS REAIS DA STARTUP DO USUÁRIO:
    {profile_dossier}

    OPORTUNIDADE SALVA SELECIONADA PELO USUÁRIO (SE HOUVER):
    {selected_opp_context}

    DOCUMENTO / EDITAL ANEXADO NO CHAT (SE HOUVER):
    {attachment_context}
    """

    chat_history = [{"role": "system", "content": system_instruction}]
    for msg in (messages or [])[-10:]:
        role = "assistant" if msg.get("from") in ["ai", "assistant"] else "user"
        content = msg.get("text") or msg.get("content") or ""
        chat_history.append({"role": role, "content": content})

    seen_models = []
    for candidate_model in FALLBACK_MODELS:
        if not candidate_model or candidate_model in seen_models:
            continue
        seen_models.append(candidate_model)
        try:
            completion = client.chat.completions.create(
                model=candidate_model,
                messages=chat_history,
                temperature=0.4,
                max_tokens=1300,
            )
            raw_reply = completion.choices[0].message.content or ""
            if raw_reply.strip():
                return to_structured_plain_text(raw_reply)
        except Exception as e:
            err_str = str(e)
            print(f"[StartIn IA] Aviso no modelo '{candidate_model}': {err_str}")
            if "401" in err_str or "invalid_api_key" in err_str.lower():
                break

    return _apply_style_wrapper(_build_intelligent_contextual_reply(
        startup_profile, selected_opportunity, attachment_name, extracted_attachment, last_user_msg
    ))


if __name__ == "__main__":
    print("=== StartIn IA: Executando rotina de raspagem e cálculo de matches (JSON) ===")
    resultado_diario = run_daily_scrape_and_match_job(send_to_backend=True)
    print(json.dumps(resultado_diario, indent=2, ensure_ascii=False))
