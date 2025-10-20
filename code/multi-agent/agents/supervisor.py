# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import os

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

from .math import math_agent
from .web_search import research_agent

llm = ChatOpenAI(
    model_name=os.environ.get("LLM_MODEL_NAME"),
    openai_api_key=os.environ.get("LLM_API_KEY"),
    openai_api_base=os.environ.get("LLM_URL") + "/v1",
)

supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()
