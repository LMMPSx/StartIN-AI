import os
import sys
import time
import threading
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Assegurar codificação UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from IA import (
    evaluate_startup_matches,
    generate_assistant_chat,
    scrape_real_web_opportunities,
    fetch_user_and_startup_from_backend,
    fetch_opportunities_from_backend,
    run_daily_scrape_and_match_job,
    client,
    MODEL_NAME,
)

# Controle do agendador diário das 08:00 da manhã
_LAST_EXECUTED_DATE: Optional[str] = None
_SCHEDULER_STOP_EVENT = threading.Event()


def _daily_08am_scheduler_loop():
    """
    Thread em background que monitora o relógio do servidor e executa
    automaticamente todos os dias às 08:00 da manhã o web scraping e
    o cálculo de matches para todas as startups, enviando o JSON para o banco.
    """
    global _LAST_EXECUTED_DATE
    print("[StartIn IA] Agendador automático diário (08:00 da manhã) iniciado.")
    while not _SCHEDULER_STOP_EVENT.is_set():
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        if now.hour == 8 and now.minute == 0 and _LAST_EXECUTED_DATE != today_str:
            _LAST_EXECUTED_DATE = today_str
            try:
                run_daily_scrape_and_match_job(send_to_backend=True)
            except Exception as e:
                print(f"[StartIn IA] Erro na rotina agendada das 08:00: {e}")
        _SCHEDULER_STOP_EVENT.wait(20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler_thread = threading.Thread(target=_daily_08am_scheduler_loop, daemon=True)
    scheduler_thread.start()
    yield
    _SCHEDULER_STOP_EVENT.set()


app = FastAPI(
    title="StartIn - AI Matchmaking & Daily 08:00 Scraper Service",
    description="Microserviço em Python integrado ao Backend Spring Boot + MySQL com scraping e cálculo diário às 08:00",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === DTOs (Pydantic Models) ===

class StartupProfileInput(BaseModel):
    id: Optional[Any] = None
    userId: Optional[Any] = None
    userName: Optional[str] = None
    userEmail: Optional[str] = None

    # 01. Identidade
    name: Optional[str] = None
    sector: Optional[str] = None
    mainArea: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    cnpj: Optional[str] = None
    website: Optional[str] = None
    foundedYear: Optional[str] = None

    # 02. Solução
    problem: Optional[str] = None
    howItWorks: Optional[str] = None
    differentiator: Optional[str] = None
    impact: Optional[str] = None
    tags: Optional[Any] = None
    technologies: Optional[Any] = None
    businessModel: Optional[str] = None

    # 03. Mercado
    targetAudience: Optional[str] = None
    mainSector: Optional[str] = None
    operationRegion: Optional[str] = None
    state: Optional[str] = None
    customerProfile: Optional[str] = None
    marketSize: Optional[str] = None
    competitors: Optional[str] = None
    mainChannel: Optional[str] = None

    # 04. Momento e Tração
    stage: Optional[str] = None
    currentStage: Optional[str] = None
    tractionCustomers: Optional[str] = None
    productMaturity: Optional[str] = None
    teamSize: Optional[str] = None
    annualRevenue: Optional[str] = None
    recentGrowth: Optional[str] = None
    investmentRaised: Optional[str] = None

    # 05. Objetivos
    mainObjective: Optional[str] = None
    investmentNeed: Optional[str] = None
    executionDeadline: Optional[str] = None
    resourceUse: Optional[str] = None
    twelveMonthGoal: Optional[str] = None
    strategicPriority: Optional[str] = None
    desiredPartners: Optional[str] = None

    # 06. Preferências de Matching
    areasOfInterest: Optional[str] = None
    opportunityTypes: Optional[str] = None
    receiveAiRecommendations: Optional[bool] = True


class OpportunityInput(BaseModel):
    id: Optional[Any] = None
    code: Optional[str] = None
    title: str
    type: Optional[str] = "EDITAL"
    shortDescription: Optional[str] = ""
    description: Optional[str] = ""
    tags: Optional[str] = ""
    url: Optional[str] = ""
    externalUrl: Optional[str] = ""
    deadline: Optional[str] = ""
    meta: Optional[str] = ""
    metaInfo: Optional[str] = ""


class MatchRequest(BaseModel):
    startup: Optional[StartupProfileInput] = None
    startupId: Optional[int] = None
    userId: Optional[int] = None
    opportunities: Optional[List[OpportunityInput]] = Field(default_factory=list)
    scrapeLiveWeb: Optional[bool] = True


class MatchReasons(BaseModel):
    perfil: str = ""
    requisitos: str = ""
    objetivo: str = ""
    contexto: str = ""


class MatchItem(BaseModel):
    opportunityId: Optional[str] = None
    opportunityTitle: str
    opportunityType: Optional[str] = "EDITAL"
    opportunityDescription: Optional[str] = ""
    opportunityMeta: Optional[str] = ""
    opportunityTags: Optional[str] = ""
    url: Optional[str] = ""
    matchScore: int
    justification: str
    recommendedAction: str
    reasons: Optional[MatchReasons] = None


class MatchResponse(BaseModel):
    startupName: str
    totalAnalyzed: int
    matches: List[MatchItem]


class ScrapeRequest(BaseModel):
    urls: Optional[List[str]] = None
    forceRefresh: Optional[bool] = False


class ChatMessage(BaseModel):
    from_: str = Field(default="user", alias="from")
    text: str

    class Config:
        populate_by_name = True


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    startup: Optional[StartupProfileInput] = None
    startupId: Optional[int] = None
    userId: Optional[int] = None


class ChatResponse(BaseModel):
    reply: str


# === Endpoints ===

@app.get("/health")
@app.get("/api/ai/health")
def health_check():
    """Verifica a saúde do microserviço, conectividade com a Groq e status do agendador das 08:00."""
    return {
        "status": "UP",
        "service": "StartIn-IA",
        "groqConnected": client is not None,
        "model": MODEL_NAME,
        "dailySchedule": "08:00",
        "lastDailyRun": _LAST_EXECUTED_DATE,
    }


@app.get("/api/ai/daily-scrape-json")
@app.post("/api/ai/daily-scrape-json")
def daily_scrape_and_match_json_endpoint():
    """
    Executa a raspagem completa de editais na web e o cálculo de matches para as startups,
    retornando o JSON estruturado para ser persistido no banco de dados MySQL.
    """
    return run_daily_scrape_and_match_job(send_to_backend=False)


@app.post("/api/ai/match", response_model=MatchResponse)
def compute_match(request: MatchRequest):
    """
    Recebe os dados completos do usuário e da startup enviados pelo Spring Boot,
    combina as oportunidades do banco MySQL com os editais reais raspados da web,
    e retorna o cálculo de aderência (0-100), justificativa e os 4 pilares.
    """
    startup_dict = request.startup.model_dump() if request.startup else {}

    if not startup_dict.get("name") or (not startup_dict.get("description") and not startup_dict.get("problem")):
        sid = request.startupId or startup_dict.get("id")
        uid = request.userId or startup_dict.get("userId")
        backend_profile = fetch_user_and_startup_from_backend(startup_id=sid, user_id=uid)
        if backend_profile:
            startup_dict = {
                **backend_profile,
                **{k: v for k, v in startup_dict.items() if v not in (None, "", [])},
            }

    if not startup_dict.get("tags") and startup_dict.get("technologies"):
        techs = startup_dict["technologies"]
        if isinstance(techs, list):
            startup_dict["tags"] = techs
        else:
            startup_dict["tags"] = [t.strip() for t in str(techs).replace("•", ",").split(",") if t.strip()]

    opps_list = []
    if request.opportunities:
        for opp in request.opportunities:
            opp_dict = opp.model_dump()
            opp_dict["id"] = str(opp_dict.get("id") or opp_dict.get("code") or "")
            opps_list.append(opp_dict)
    else:
        opps_list = fetch_opportunities_from_backend()

    if request.scrapeLiveWeb is not False:
        scraped_web = scrape_real_web_opportunities()
        existing_titles = {str(o.get("title", "")).strip().lower() for o in opps_list}
        for web_opp in scraped_web:
            if web_opp["title"].strip().lower() not in existing_titles:
                opps_list.append(web_opp)
                existing_titles.add(web_opp["title"].strip().lower())

    try:
        raw_result = evaluate_startup_matches(startup_dict, opps_list)
        raw_matches = raw_result.get("matches", [])

        formatted_matches: List[MatchItem] = []
        for m in raw_matches:
            opp_id = str(m.get("opportunity_id") or m.get("opportunityId") or "")
            opp_title = m.get("opportunity_title") or m.get("opportunityTitle") or "Oportunidade"
            opp_type = m.get("opportunity_type") or m.get("opportunityType") or "EDITAL"
            opp_desc = m.get("opportunity_description") or m.get("opportunityDescription") or ""
            opp_meta = m.get("opportunity_meta") or m.get("opportunityMeta") or "Inscrições abertas · Brasil"
            opp_tags = m.get("opportunity_tags") or m.get("opportunityTags") or "Inovação • Tecnologia"
            url = m.get("url", "")
            score = int(m.get("match_score") or m.get("matchScore") or 75)
            justification = m.get("justification", "Perfil compatível com os critérios da oportunidade.")
            rec_action = m.get("recommended_action") or m.get("recommendedAction") or "Submeter candidatura."

            raw_reasons = m.get("reasons") or {}
            reasons = MatchReasons(
                perfil=raw_reasons.get("perfil", "Alinhamento com segmento e mercado"),
                requisitos=raw_reasons.get("requisitos", "Requisitos técnicos atendidos"),
                objetivo=raw_reasons.get("objetivo", "Objetivo estratégico convergente"),
                contexto=raw_reasons.get("contexto", "Modelo operacional compatível"),
            )

            formatted_matches.append(
                MatchItem(
                    opportunityId=opp_id,
                    opportunityTitle=opp_title,
                    opportunityType=opp_type,
                    opportunityDescription=opp_desc,
                    opportunityMeta=opp_meta,
                    opportunityTags=opp_tags,
                    url=url,
                    matchScore=score,
                    justification=justification,
                    recommendedAction=rec_action,
                    reasons=reasons,
                )
            )

        return MatchResponse(
            startupName=startup_dict.get("name") or "Startup",
            totalAnalyzed=len(opps_list),
            matches=formatted_matches,
        )
    except Exception as e:
        print(f"Erro no processamento do match: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno de matchmaking: {str(e)}")


@app.post("/api/ai/scrape-opportunities")
def scrape_opportunities_endpoint(request: ScrapeRequest):
    """Endpoint para raspar editais e chamadas reais de portais oficiais em JSON."""
    extracted_items = scrape_real_web_opportunities(
        urls=request.urls,
        force_refresh=bool(request.forceRefresh),
    )
    return {"count": len(extracted_items), "opportunities": extracted_items}


@app.post("/api/ai/chat", response_model=ChatResponse)
def chat_assistant_endpoint(request: ChatRequest):
    """
    Endpoint do Assistente Virtual:
    - Responde SOMENTE sobre o projeto StartIn (oportunidades, matches e perfil da startup).
    - Formato estritamente texto plano (sem markdown) e sucinto.
    """
    msgs = [{"from": m.from_, "text": m.text} for m in request.messages]
    startup_dict = request.startup.model_dump() if request.startup else {}
    if request.startupId and not startup_dict.get("id"):
        startup_dict["id"] = request.startupId
    if request.userId and not startup_dict.get("userId"):
        startup_dict["userId"] = request.userId

    reply = generate_assistant_chat(msgs, startup_dict)
    return ChatResponse(reply=reply)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Iniciando StartIn AI Service na porta {port} (Rotina diária agendada para 08:00)...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
