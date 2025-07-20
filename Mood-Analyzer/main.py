from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig

import os
import chainlit as cl 
from dotenv import load_dotenv

load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")

external_client=AsyncOpenAI(
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

# Agent 1 Mood Detector
mood_detector= Agent(
    name="Mood Detector",
    instructions="""
You are a mood detector.
Based on user's message, respond with only one mood word from:
happy, sad, neutral, depressed.
No explanations. Just the mood word.
"""
)

# Agent 2 Activity Suggester
activity_suggestor= Agent(
    name="Activity Suggester",
    instructions="""
User is feeling sad or depressed.
Suggest one short and helpful activity to lift their mood.
Be friendly and concise.
"""
)

@cl.on_chat_start
async def on_start():
    await cl.Message(content="Hello how you are feeling today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content

    # Mood Detection
    result1 = await Runner.run(mood_detector, input=user_input, run_config=config)
    mood = result1.final_output.strip().lower()

    await cl.Message(content=f"**Detected Mood:** `{mood}`").send()

    # Suggestion only if mood is sad or depressed
    if mood in ["sad", "depressed"]:
        result2 = await Runner.run(activity_suggestor, input=f"My mood is {mood}", run_config=config)
        activity = result2.final_output.strip()
        await cl.Message(content=f" **Suggested Activity:** {activity}").send()
    else:
        await cl.Message(content="You seem to be doing great! Keep smiling and enjoy your day!").send()
