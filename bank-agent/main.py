from agents import Agent, Runner, OpenAIChatcompletionsModel, input_guardrail, output_guardrail, set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import RunContextWrapper, TResponseInputItem, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
import chainlit as cl 
import os

# load Enviroment variables
load_dotenv()
set_tracing_disabled(disabled=True)

# Gemini Api Key Setup 
gemini_api_key = os.getenv("Gemini_api_key")
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model =OpenAIChatcompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Input Guardrails Schema
class Bankinputcheck(BaseModel):
    is_bank_related: bool
    reasoning: str

input_guardrails_agent = Agent(
    name="Input Guardrail Checker",
    instructions="Check if the question is about banking. Return true if it is; false otherwise.",
    model=model,
    output_type=Bankinputcheck
)

@input_guardrail
async def validate_banking_input(ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrails_agent, input)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_banking_related
    )

# Output Guardrails Schema
class output_safety_check(BaseModel):
    is_safe: bool
    resoning: str

output_guardrails_agent =Agent(
    name="Output Guardrails checker",
    instructions="Check if the agent's response contains personal or unsafe information. Return false if unsafe.",
    model=model,
    output_type=output_safety_check
)

@output_guardrail
async def validate_output_safety(ctx: RunContextWrapper, agent: Agent, output: str) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrails_agent, output)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_safe
    )

# handoff Agent
account_agent = Agent(
    name="Account Agent",
    instructions="You are an account management agent. Answer only about account balances, statements, and account details.",
    model=model
)

loan_agent = Agent(
    name="loan agent",
    instructions="you are a loan agent. Answer only about loan balances, statements, and loan details.",
    model=model
)

# Main Banking Agent
banking_agent = Agent(
    name="Banking Agent",
    instructions="You are a banking assistant. Only answer banking-related queries. Route to correct department.",
    model=model,
    input_guardrails=[validate_banking_input],
    output_guardrails=[validate_output_safety],
    handoffs=[account_agent, loan_agent]
)

# Chainlit Handlers
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="💳 Welcome to the Banking Assistant! How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    try:
        result = await Runner.run(banking_agent, input=message.content)
        await cl.Message(content=str(result.final_output)).send()
    except InputGuardrailTripwireTriggered:
        await cl.Message(await cl.Message(content="Please Ask Banking Questions").send())
    
