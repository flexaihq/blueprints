# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import getpass
import os

from agents.supervisor import supervisor
from langchain_core.messages import convert_to_messages


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("LLM_API_KEY")
_set_if_undefined("LLM_URL")
_set_if_undefined("LLM_MODEL_NAME")
_set_if_undefined("TAVILY_API_KEY")


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


while True:
    user_question = input("Please enter your question: ")

    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_question,
                }
            ]
        },
    ):
        if not ("supervisor" in chunk and chunk["supervisor"] is None):
            pretty_print_messages(chunk, last_message=True)

    final_message_history = chunk["supervisor"]["messages"]
