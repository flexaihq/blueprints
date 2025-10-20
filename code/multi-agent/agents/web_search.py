# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import os

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

web_search = TavilySearch(max_results=3)


llm = ChatOpenAI(
    model_name=os.environ.get("LLM_MODEL_NAME"),
    openai_api_key=os.environ.get("LLM_API_KEY"),
    openai_api_base=os.environ.get("LLM_URL") + "/v1",
)

research_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)
