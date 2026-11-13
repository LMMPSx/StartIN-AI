"""
Ponto de entrada serverless do Vercel para o microserviço de IA (FastAPI).
Todas as rotas (/, /health, /api/ai/...) são reescritas para esta função pelo vercel.json.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server import app  # noqa: E402,F401
