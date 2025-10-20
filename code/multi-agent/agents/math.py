# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(
    model_name=os.environ.get("LLM_MODEL_NAME"),
    openai_api_key=os.environ.get("LLM_API_KEY"),
    openai_api_base=os.environ.get("LLM_URL") + "/v1",
)


def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)
