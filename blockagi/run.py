import os
from blockagi.chains import BlockAGIChain
from blockagi.schema import Findings
from blockagi.tools import (
    DDGSearchAnswerTool,
    DDGSearchLinksTool,
    GoogleSearchLinksTool,
    VisitWebTool,
)
from langchain_groq import ChatGroq


def run_blockagi(
    agent_role,
    groq_api_key,
    groq_model,
    resource_pool,
    objectives,
    blockagi_callback,
    llm_callback,
    iteration_count,
):
    tools = []
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        tools.append(GoogleSearchLinksTool(resource_pool))

    tools.extend(
        [
            DDGSearchAnswerTool(),
            DDGSearchLinksTool(resource_pool),
            VisitWebTool(resource_pool),
        ]
    )

    blockagi_callback.on_log_message(
        f"Using {len(tools)} tools:\n"
        + "\n".join(
            [f"{idx+1}. {t.name} - {t.description}" for idx, t in enumerate(tools)]
        )
    )

    # Menggunakan GROQ API sebagai pengganti OpenAI
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=groq_model,
        temperature=0.8,
        streaming=True,
    )
    
    blockagi_callback.on_log_message(f"GROQ client initialized with model: {groq_model}")

    inputs = {
        "objectives": objectives,
        "findings": Findings(
            narrative="Nothing",
            remark="",
            generated_objectives=[],
        ),
    }

    BlockAGIChain(
        iteration_count=iteration_count,
        agent_role=agent_role,
        llm=llm,
        tools=tools,
        resource_pool=resource_pool,
        callbacks=[blockagi_callback],
    )(inputs=inputs)
