import json
from typing import List, Dict, Any
from langchain.chat_models.base import BaseChatModel
from langchain.tools.base import BaseTool
from langchain.schema import HumanMessage, SystemMessage
from blockagi.chains.base import CustomCallbackLLMChain
from blockagi.utils import to_json_str, format_objectives

from blockagi.schema import Objective, Findings, Narrative


class EvaluateChain(CustomCallbackLLMChain):
    agent_role: str = "a Research Assistant"
    tools: List[BaseTool]

    @property
    def input_keys(self) -> List[str]:
        return [
            "objectives",  # Primary input
            "findings",  # Previous findings
            "narrative",  # Narrate    -> Evaluate
        ]

    @property
    def output_keys(self) -> List[str]:
        return [
            # Feedback to next iteration
            "updated_findings",  # Evaluate   -> Plan
            "updated_objectives",  # Evaluate   -> Plan
        ]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        objectives: List[Objective] = inputs["objectives"]
        findings: Findings = inputs["findings"]
        narrative: Narrative = inputs["narrative"]

        self.fire_log("Evaluating the narrative for the next iteration")
        response_format = {
            "updated_findings": {
                "generated_objectives": [
                    Objective(
                        topic="additional objective that helps achieve the user objectives",
                        expertise="a new float value in [0, 1] range indicating the expertise of this objective",
                    ),
                    "... include all generated objectives",
                ],
                "remark": "a note to the next iteration of BlockAGI to help it improve",
            },
            "updated_objectives": [
                Objective(
                    topic="same as the user objectives",
                    expertise="a new float value in [0, 1] range indicating the expertise of this objective",
                ),
                "... include all objectives",
            ],
        }

        messages = [
            SystemMessage(
                content=f"You are {self.agent_role}. "
                "Your job is to evaluate YOUR FINDING become expert in the primary goals "
                "under OBJECTIVES and the secondary goals under GENERATED_OBJECTIVES. "
                "Take into account the limitation of all the tools available to you."
                "\n\n"
                "## USER OBJECTIVES:\n"
                f"{format_objectives(objectives)}\n\n"
                "## GENERATED OBJECTIVES:\n"
                f"{format_objectives(findings.generated_objectives)}\n\n"
                "## REMARK:\n"
                f"{findings.remark}\n\n"
                "You should ONLY respond in the JSON format as described below\n"
                "## RESPONSE FORMAT:\n"
                f"{to_json_str(response_format)}"
            ),
            HumanMessage(
                content="You just finished a research iteration and formulated a FINDING below.\n"
                "## YOUR FINDINGS:\n"
                "```\n"
                f"{narrative.markdown}\n"
                "```\n\n"
                "# YOUR TASK:\n"
                "Give a thorough evaluation of your work and plan to become a better expert. "
                "Your evaluation should include:\n"
                "- Modified up to 1 new GENERATED OBJECTIVE to help yourself become an "
                "expert and answer USER OBJECTIVES with further research. Do not modify the USER OBJECTIVES.\n"
                "- A remark to help the next iteration of BlockAGI improve. Be critical and suggest "
                "only concise and helpful feedback for the AI agent.\n"
                "- A new expertise weight between (0 and 1) of all the OBJECTIVES. "
                "If the goal is close to being met, its expertise should be higher."
                "\n\n"
                "# YOUR TASK:\n"
                "Respond using ONLY the format specified above:"
            ),
        ]

        response = self.retry_llm(messages)

        # Clean and parse response content
        content = response.content.strip()
        
        # Try to extract JSON from response
        try:
            # First try direct parsing
            result = json.loads(content)
        except json.JSONDecodeError as e:
            self.fire_log(f"JSON parsing failed: {e}")
            self.fire_log(f"Response content: {content[:200]}...")
            
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    self.fire_log("Successfully extracted JSON from code block")
                except json.JSONDecodeError:
                    self.fire_log("Failed to parse JSON from code block")
                    raise
            else:
                # Try to find JSON-like content
                brace_start = content.find('{')
                brace_end = content.rfind('}')
                if brace_start != -1 and brace_end != -1:
                    try:
                        json_content = content[brace_start:brace_end + 1]
                        result = json.loads(json_content)
                        self.fire_log("Successfully extracted JSON from content")
                    except json.JSONDecodeError:
                        self.fire_log("Failed to parse extracted JSON content")
                        raise
                else:
                    raise json.JSONDecodeError("No JSON content found in response", content, 0)

        updated_findings = Findings(
            generated_objectives=[
                Objective(
                    topic=obj["topic"],
                    expertise=obj["expertise"],
                )
                for obj in result["updated_findings"]["generated_objectives"]
            ],
            remark=result["updated_findings"]["remark"],
            narrative=narrative.markdown,
        )

        self.fire_log(f'Agent\'s remark: "{updated_findings.remark}"')
        updated_objectives = [
            Objective(
                topic=obj["topic"],
                expertise=obj["expertise"],
            )
            for obj in result["updated_objectives"]
        ]

        return {
            "updated_findings": updated_findings,
            "updated_objectives": updated_objectives,
        }
