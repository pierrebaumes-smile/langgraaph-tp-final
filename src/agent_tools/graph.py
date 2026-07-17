from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from agent_tools.model import get_model
from agent_tools.tools import run_python, search_docs

TOOLS = [search_docs, run_python]

SENSITIVE_TOOLS = {"run_python"}

SYSTEM_PROMPT = (
    "Tu es un assistant développeur LangGraph. Pour toute question sur "
    "l'API ou les concepts LangGraph, cherche d'abord dans la documentation "
    "avec l'outil search_docs avant de répondre. Si l'utilisateur te demande "
    "d'écrire et de tester du code, propose le code puis appelle l'outil "
    "run_python pour l'exécuter — son exécution sera soumise à validation "
    "humaine avant de tourner réellement."
)


class AgentState(MessagesState):
    validated: bool | None


def agent_node(state: AgentState) -> dict:
    model = get_model().bind_tools(TOOLS)
    messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_after_agent(state: AgentState) -> Literal["validate", "tools", "__end__"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return END
    if any(call["name"] in SENSITIVE_TOOLS for call in tool_calls):
        return "validate"
    return "tools"


def validate_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    decision = interrupt(
        {
            "question": "Approuvez-vous l'exécution de ces outils ?",
            "tool_calls": [
                {"name": call["name"], "args": call["args"]}
                for call in last_message.tool_calls
            ],
        }
    )
    return {"validated": bool(decision)}


def approval_route(state: AgentState) -> Literal["tools", "reject"]:
    return "tools" if state.get("validated") else "reject"


def reject_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    rejections = [
        ToolMessage(
            content="Exécution refusée par l'utilisateur.",
            tool_call_id=call["id"],
        )
        for call in last_message.tool_calls
    ]
    return {"messages": rejections, "validated": False}


def _builder() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("validate", validate_node)
    builder.add_node("reject", reject_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["validate", "tools", END])
    builder.add_conditional_edges("validate", approval_route, ["tools", "reject"])
    builder.add_edge("tools", "agent")
    builder.add_edge("reject", "agent")

    return builder


def build_graph():
    return _builder().compile(checkpointer=MemorySaver())

graph = _builder().compile()
