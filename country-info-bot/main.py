from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig, function_tool

import os
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("gemini_api_key")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# -------------------- Tool 1: Capital --------------------
@function_tool
def get_capital(country: str) -> str:
    """Return the capital of a country.

    Args:
        country: The name of the country.
    """
    capitals = {
        "pakistan": "Islamabad",
        "india": "New Delhi",
        "japan": "Tokyo",
        "germany": "Berlin",
        "france": "Paris",
        "canada": "Ottawa",
        "australia": "Canberra",
    }
    return capitals.get(country.lower(), "Capital not found.")

# -------------------- Tool 2: Language --------------------
@function_tool
def get_language(country: str) -> str:
    """Return the official language of a country.

    Args:
        country: The name of the country.
    """
    languages = {
        "pakistan": "Urdu",
        "india": "Hindi",
        "japan": "Japanese",
        "germany": "German",
        "france": "French",
        "canada": "English and French",
        "australia": "English",
    }
    return languages.get(country.lower(), "Language not found.")

# -------------------- Tool 3: Population --------------------
@function_tool
def get_population(country: str) -> str:
    """Return the approximate population of a country.

    Args:
        country: The name of the country.
    """
    populations = {
        "pakistan": "240 million",
        "india": "1.4 billion",
        "japan": "125 million",
        "germany": "83 million",
        "france": "67 million",
        "canada": "39 million",
        "australia": "26 million",
    }
    return populations.get(country.lower(), "Population not found.")

# -------------------- Orchestrator Agent --------------------
orchestrator = Agent(
    name="Country Info Bot",
    instructions=(
        "You are a helpful assistant. When given a country name, use the tools to find "
        "its capital city, official language, and population. Combine and present them clearly."
    ),
    tools=[get_capital, get_language, get_population],
)

# -------------------- Chainlit Message Handler --------------------
@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()
    result = await Runner.run(orchestrator, input=user_input, run_config=config)
    await cl.Message(content=result.final_output).send()