import os
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from ..control.task_structures import InspectionPacket, FailureCode
from tools.logging.task_logger import log_error, log_warning
from ..utils.config import Config
from tools.vision.upload_image import upload_file_and_get_url

@dataclass
class Hazard:
    type: str
    severity: str  # "low", "medium", "high", "critical"
    why: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "why": self.why
        }

@dataclass
class InspectionReport:
    what_happened: str
    hazards: List[Hazard]
    optional_extra_suggestions: List[Dict[str, Any]]
    verdict_hint: str  # "SUCCESS", "FAIL", "UNCERTAIN"
    confidence: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "what_happened": self.what_happened,
            "hazards": [h.to_dict() for h in self.hazards],
            "optional_extra_suggestions": self.optional_extra_suggestions,
            "verdict_hint": self.verdict_hint,
            "confidence": self.confidence,
            "error": self.error
        }

class VLMInspector:
    """
    Client tool to inspect an InspectionPacket using a VLM.
    Converts raw evidence into a structured diagnostic report.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("Zhipu_real_demo_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
        self.model = model or Config.VLM_NAME or "qwen-vl-max"
        self.provider = "dashscope" if self.api_key and self.api_key.startswith("sk-") else "zhipu"
        if not self.api_key:
            self.provider = "stub"

    def inspect(self, packet: InspectionPacket) -> InspectionReport:
        """
        Analyze the inspection packet and return a structured report.
        Gracefully degrades if VLM is unavailable or fails.
        """
        try:
            if self.provider == "stub":
                return self._stub_inspect(packet)
            
            return self._vlm_inspect(packet)
        except Exception as e:
            log_error(f"❌ [VLMInspector] Inspection failed: {e}")
            return InspectionReport(
                what_happened="Inspection process crashed.",
                hazards=[],
                optional_extra_suggestions=[],
                verdict_hint="UNCERTAIN",
                confidence=0.0,
                error=str(e)
            )

    def _stub_inspect(self, packet: InspectionPacket) -> InspectionReport:
        """Fallback logic when no VLM is configured."""
        status = packet.exec_result.get("status", "unknown")
        failure_code = packet.exec_result.get("failure_code")
        
        verdict = "SUCCESS" if status == "success" else "FAIL"
        if status == "unknown":
            verdict = "UNCERTAIN"

        what = f"Action '{packet.skill_name}' finished with status '{status}'."
        if failure_code:
            what += f" Failure code: {failure_code}."
        
        hazards = []
        if verdict == "FAIL":
            hazards.append(Hazard(type="execution_failure", severity="medium", why=f"Skill failed: {packet.exec_result.get('reason')}"))

        return InspectionReport(
            what_happened=what,
            hazards=hazards,
            optional_extra_suggestions=[],
            verdict_hint=verdict,
            confidence=0.5, # Low confidence for stub
            error="VLM not configured (stub mode)"
        )

    def _build_prompt(self, packet: InspectionPacket) -> str:
        # Extract key info for prompt
        skill = packet.skill_name
        args = packet.skill_args
        result = packet.exec_result
        metrics = packet.raw_metrics or {}
        verifier = packet.verifier_outputs or {}
        
        prompt = f"""
You are a Robot Diagnostic Agent. Analyze the following execution report.

CONTEXT:
- Action: {skill}
- Args: {json.dumps(args)}
- Result Status: {result.get('status')}
- Failure Code: {result.get('failure_code')}
- Reason: {result.get('reason')}

EVIDENCE:
- Metrics: {json.dumps(metrics)}
- Verifier Findings: {json.dumps(verifier)}

TASK:
1. Describe what happened based strictly on the evidence. Quote specific metrics (e.g. "gripper_width=5mm").
2. Identify any hazards or risks (e.g. "collision", "object_slip", "blind_spot").
3. Provide a verdict hint (SUCCESS/FAIL/UNCERTAIN).
4. Suggest any extra recovery steps if obvious.

OUTPUT JSON FORMAT:
{{
  "what_happened": "...",
  "hazards": [
    {{"type": "...", "severity": "low|medium|high", "why": "..."}}
  ],
  "optional_extra_suggestions": [
    {{"action": "...", "reason": "..."}}
  ],
  "verdict_hint": "SUCCESS|FAIL|UNCERTAIN",
  "confidence": 0.0-1.0
}}
"""
        return prompt.strip()

    def _vlm_inspect(self, packet: InspectionPacket) -> InspectionReport:
        # Prioritize post-execution observation if available
        image_url = None
        if packet.post_execution_observation:
            image_url = packet.post_execution_observation.get("annotated_url") or \
                        packet.post_execution_observation.get("url") or \
                        packet.post_execution_observation.get("image_path")
        
        # Fallback to artifacts (usually pre-action observation)
        if not image_url:
            image_url = packet.artifacts.get("annotated_url") or packet.artifacts.get("image_path")

        prompt = self._build_prompt(packet)
        
        response_text = "{}"
        try:
            if self.provider == "dashscope":
                import dashscope
                messages = [{"role": "user", "content": [{"text": prompt}]}]
                if image_url:
                    # Check if local path or URL
                    if os.path.exists(image_url) and not image_url.startswith("http"):
                         try:
                             oss_url = upload_file_and_get_url(
                                 api_key=self.api_key,
                                 model_name=self.model,
                                 file_path=image_url,
                             )
                             messages[0]["content"].insert(0, {"image": oss_url})
                         except Exception as e:
                             log_warning(f"⚠️ [VLMInspector] Failed to upload local image: {e}")
                    else:
                         messages[0]["content"].insert(0, {"image": image_url})

                response = dashscope.MultiModalConversation.call(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key
                )
                if response.status_code == 200:
                    response_text = response.output.choices[0].message.content[0]["text"]
                else:
                    raise RuntimeError(f"Dashscope error: {response.message}")

            elif self.provider == "zhipu":
                from zhipuai import ZhipuAI
                client = ZhipuAI(api_key=self.api_key)
                content = [{"type": "text", "text": prompt}]
                if image_url and image_url.startswith("http"):
                     content.append({"type": "image_url", "image_url": {"url": image_url}})
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}]
                )
                response_text = response.choices[0].message.content

            # Parse JSON
            data = self._parse_json(response_text)
            
            hazards = [Hazard(**h) for h in data.get("hazards", [])]
            
            return InspectionReport(
                what_happened=data.get("what_happened", "Analysis failed to produce description."),
                hazards=hazards,
                optional_extra_suggestions=data.get("optional_extra_suggestions", []),
                verdict_hint=data.get("verdict_hint", "UNCERTAIN"),
                confidence=float(data.get("confidence", 0.5)),
                error=None
            )

        except Exception as e:
            log_warning(f"⚠️ [VLMInspector] VLM call failed, falling back to stub: {e}")
            return self._stub_inspect(packet)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except:
            return {}
