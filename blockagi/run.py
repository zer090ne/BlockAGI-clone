import os
from blockagi.chains import BlockAGIChain
from blockagi.schema import Findings
from blockagi.tools import (
    DDGSearchAnswerTool,
    DDGSearchLinksTool,
    GoogleSearchLinksTool,
    VisitWebTool,
)
from groq import Groq


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
    llm = Groq(
        api_key=groq_api_key,
        model=groq_model,  # misal: "llama3-8b-8192", "mixtral-8x7b-32768"
        temperature=0.8,
        streaming=True,
        callbacks=[llm_callback],
    )

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
