from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from agent_tools.graph import build_graph


def _drive(graph, stream_input, config, printed: set[str]) -> object | None:
    interrupt_payload = None
    for chunk in graph.stream(stream_input, config=config, stream_mode="values"):
        if "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"][0].value
            continue
        messages = chunk.get("messages")
        if messages and messages[-1].id not in printed:
            messages[-1].pretty_print()
            printed.add(messages[-1].id)
    return interrupt_payload


def run_turn(graph, config, user_input: str, auto_approve: bool | None = None) -> None:
    printed: set[str] = set()
    payload = _drive(graph, {"messages": [HumanMessage(user_input)]}, config, printed)

    while payload is not None:
        print(f"\n[VALIDATION REQUISE] {payload}\n")
        if auto_approve is None:
            answer = input("Approuvez-vous ? (o/n) ").strip().lower()
            decision = answer in ("o", "oui", "y", "yes")
        else:
            decision = auto_approve
            print(f"(auto) -> {'approuvé' if decision else 'refusé'}")
        payload = _drive(graph, Command(resume=decision), config, printed)


def run_demo(graph) -> None:
    config = {"configurable": {"thread_id": f"demo-{uuid.uuid4()}"}}

    print("\n===== Scénario 1 : question qui déclenche une recherche doc =====")
    run_turn(
        graph,
        config,
        "Comment fonctionne le checkpointer MemorySaver dans LangGraph ?",
    )

    print("\n===== Scénario 2 : exécution de code, APPROUVÉE =====")
    run_turn(
        graph,
        config,
        auto_approve=True,
    )

    print("\n===== Scénario 3 : exécution de code, REFUSÉE =====")
    run_turn(
        graph,
        config,
        auto_approve=False,
    )


def run_interactive(graph) -> None:
    config = {"configurable": {"thread_id": f"cli-{uuid.uuid4()}"}}
    print("Assistant développeur LangGraph — tapez 'exit' pour quitter.\n")
    while True:
        try:
            user_input = input("Vous: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        run_turn(graph, config, user_input)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant développeur LangGraph")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Exécute un scénario scripté (recherche doc + exécution approuvée + exécution refusée)",
    )
    args = parser.parse_args()

    graph = build_graph()

    if args.demo:
        run_demo(graph)
    else:
        run_interactive(graph)


if __name__ == "__main__":
    main()
