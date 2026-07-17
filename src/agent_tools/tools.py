"""Outils fournis pour le TP final — RAG hybride + exécution Python."""
import subprocess

from langchain_core.tools import tool

from agent_tools.rag import get_index


@tool
def search_docs(query: str) -> str:
    """Cherche dans la documentation technique LangGraph.

    Utilise un index hybride (BM25 + embeddings sémantiques)
    sur les pages officielles extraites de docs.langchain.com.
    Les résultats incluent leur source pour vérification.
    """
    try:
        index = get_index()
        results = index.search(query, k=5)
        if not results:
            return "Aucun résultat trouvé dans la documentation."

        parts = []
        max_score = max((s for _, s in results), default=1)
        for i, (chunk, score) in enumerate(results, 1):
            pct = score / max_score
            header = f"[{chunk.title}] (pertinence: {pct:.0%}, rang #{i})"
            text = chunk.text[:1200].strip()
            parts.append(f"{header}\n{text}")
        return "\n\n---\n\n".join(parts)

    except Exception as e:
        return f"Erreur de recherche documentaire : {e}"


import re

def _extract_code(raw: str) -> str:
    """Extrait le code d'un bloc markdown triple backticks si présent."""
    match = re.search(r"```(?:python)?\n(.+?)\n```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


@tool
def run_python(code: str) -> str:
    """Exécute du code Python et retourne la sortie.

    Le code est exécuté dans un sous-processus isolé. 
    Utilisé pour tester des snippets.
    """
    clean = _extract_code(code)
    if not clean:
        return "ERREUR : code vide"

    try:
        result = subprocess.run(
            ["python3", "-c", clean],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[code retour: {result.returncode}]"
        return output.strip() or "(rien — le code n'a rien affiché)"
    except subprocess.TimeoutExpired:
        return f"ERREUR : le code a dépassé 15 secondes"
    except Exception as e:
        return f"ERREUR : {e}"
