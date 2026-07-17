# TP Final — Assistant Développeur LangGraph

## Objectif

Assistant développeur LangGraph : l'agent cherche dans la documentation officielle (RAG), propose du code, et demande une validation humaine avant d'exécuter quoi que ce soit.

Deux outils sont fournis, prêts à l'emploi (`src/agent_tools/tools.py`) :

- `search_docs` — recherche hybride (BM25 + embeddings) dans `docs/`
- `run_python` — exécution de code Python en sous-processus, avec timeout

Le graphe (`src/agent_tools/graph.py`) implémente l'état, le routage, la validation humaine et la boucle agent/outils.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`search_docs` tourne en local (BM25 + `sentence-transformers`), aucune clé API n'est nécessaire pour les outils.

## Configuration

Le modèle de l'agent utilise **Groq** (`langchain-groq`). Copiez `.env.example` en `.env` et renseignez vos clés :

```bash
cp .env.example .env
```

```dotenv
# --- Modèle de l'agent (Groq) ---
GROQ_API_KEY=...
GROQ_MODEL_NAME=qwen/qwen3-32b   # voir https://console.groq.com/docs/models

# --- Tracing LangSmith (optionnel mais recommandé) ---
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=tp-final-langgraph-assistant
```

`qwen/qwen3-32b` est le modèle par défaut : c'est celui qui s'est montré le plus fiable sur Groq pour le tool-calling multi-tours pendant le développement de ce TP (`llama-3.3-70b-versatile` et `llama-3.1-8b-instant` produisaient par moments des function-calls mal formés, `openai/gpt-oss-120b` hallucinait des outils hors de la liste fournie). Adaptez `GROQ_MODEL_NAME` selon vos besoins/quota.

Pour utiliser un autre fournisseur, remplacez `get_model()` dans [src/agent_tools/model.py](src/agent_tools/model.py) par le chat model de votre choix (ex. `langchain_openai.ChatOpenAI`, `langchain_anthropic.ChatAnthropic`) — le reste du graphe est indépendant du fournisseur.

## Architecture du graphe

```
START ─▶ agent ─▶ route_after_agent ─┬─▶ validate ─▶ approval_route ─┬─▶ tools ─┐
                                      │                               └─▶ reject ┤
                                      ├─▶ tools ──────────────────────────────────┤
                                      └─▶ END                                     │
                                                                                   ▼
                                                                                 agent (boucle)
```

- **État** — `AgentState(MessagesState)` avec un champ `validated: bool | None`.
- **`agent_node`** — invoque le modèle Groq lié aux outils (`bind_tools`).
- **`route_after_agent`** — routage par nom d'outil : si le dernier message contient un appel à un outil *sensible* (`run_python`) → `validate` ; s'il contient un autre appel d'outil (`search_docs`) → `tools` directement ; sinon → `END`.
- **`validate_node`** — appelle `interrupt(...)` avec le détail des appels d'outils proposés, suspend le graphe et attend la décision humaine (`Command(resume=True/False)`). Stocke le résultat dans `validated`.
- **`approval_route`** — `tools` si `validated` est vrai, sinon `reject`.
- **`reject_node`** — construit un `ToolMessage` (avec le `tool_call_id` correspondant) pour chaque appel d'outil refusé, sans jamais exécuter le code.
- **`tools`** — `ToolNode([search_docs, run_python])` du prébuilt LangGraph.
- La boucle se referme avec les arêtes `tools → agent` et `reject → agent`, pour que le modèle puisse réagir au résultat (ou au refus).
- Compilé avec `MemorySaver()` : chaque conversation (`thread_id`) est persistée, ce qui permet à `interrupt()`/`Command(resume=...)` de fonctionner et de reprendre après coupure.

## Exemple d'utilisation

```bash
python main.py            # chat interactif, approbation via input()
python main.py --demo     # scénario scripté (voir ci-dessous)
```

Utilisation programmatique :

```python
import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from agent_tools.graph import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# 1) Question qui déclenche une recherche documentaire (pas de validation nécessaire)
for chunk in graph.stream(
    {"messages": [HumanMessage("Comment fonctionne le checkpointer MemorySaver ?")]},
    config=config,
    stream_mode="values",
):
    if "messages" in chunk:
        chunk["messages"][-1].pretty_print()

# 2) Question qui déclenche une exécution de code : le graphe s'arrête sur interrupt()
for chunk in graph.stream(
    {"messages": [HumanMessage("Calcule la somme des 10 premiers nombres premiers en Python.")]},
    config=config,
    stream_mode="values",
):
    if "__interrupt__" in chunk:
        print(chunk["__interrupt__"][0].value)  # -> détail de l'appel d'outil à valider

# 3a) Reprise avec approbation
for chunk in graph.stream(Command(resume=True), config=config, stream_mode="values"):
    ...

# 3b) Ou reprise avec refus
# for chunk in graph.stream(Command(resume=False), config=config, stream_mode="values"):
#     ...
```

### Scénario `--demo`

`python main.py --demo` déroule automatiquement les trois cas requis, sur un même `thread_id` :

1. **Recherche documentaire** — « Comment fonctionne le checkpointer MemorySaver dans LangGraph ? » → `search_docs` s'exécute directement (outil non sensible), pas d'interruption.
2. **Exécution de code, approuvée** — « Écris et exécute du code Python qui calcule la somme des 10 premiers nombres premiers. » → le graphe s'arrête sur `interrupt()`, la validation est auto-approuvée, `run_python` s'exécute et l'agent commente le résultat (`129`).
3. **Exécution de code, refusée** — « Exécute du code Python qui affiche le contenu du dossier courant avec `os.listdir('.')`. » → le graphe s'arrête sur `interrupt()`, la validation est refusée, `reject_node` renvoie un `ToolMessage` de refus et l'agent explique que l'exécution n'a pas eu lieu.

## Traçabilité (LangSmith)

Avec `LANGSMITH_TRACING=true` et une clé valide, chaque nœud du graphe (`agent`, `route_after_agent`, `validate`, `approval_route`, `tools`, `reject`) apparaît comme un run dans le projet LangSmith configuré (`LANGSMITH_PROJECT`), y compris la pause/reprise autour de `interrupt()`.

## Critères d'acceptation

- ✅ **Correct** : le graphe compile, l'agent cherche dans la doc et exécute du code.
- ✅ **Persistant** : `MemorySaver` sauvegarde l'état par `thread_id` ; l'interruption via `interrupt()` et la reprise via `Command(resume=...)` fonctionnent (approbation et refus testés).
- ✅ **Traçable** : trace LangSmith complète (agent → validate → tools/reject → agent).
- ✅ **Documenté** : ce README (installation, configuration, exemple d'utilisation).

## Rendu

Formulaire de soumission : https://forms.gle/KQ1C932dtEVFReJe9
