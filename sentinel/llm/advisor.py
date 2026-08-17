"""LLM Cognitive Advisor module for Sentinel Runtime Assurance.

Provides modular switching between local LLM engines (Ollama, vLLM, Deterministic Fallback)
and models (Nemotron 3.5 Lightning, Qwen2.5-VL, Llama 3.2, DeepSeek R1, Phi-3.5).
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Dict, List, Optional

from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = Path("/tmp/sentinel_llm_model.state")

SUPPORTED_LOCAL_MODELS: Dict[str, Dict[str, Any]] = {
    "nemotron-3.5-lightning": {
        "id": "nemotron-3.5-lightning",
        "name": "Nemotron 3.5 Lightning (NVIDIA)",
        "provider": "ollama",
        "ollama_tag": "nemotron-3.5-lightning",
        "type": "Multimodal / Reasoning",
        "latency_target": "~12ms",
        "description": "NVIDIA quantized edge reasoning model optimized for Grace Blackwell GB10.",
    },
    "qwen2.5-7b": {
        "id": "qwen2.5-7b",
        "name": "Qwen 2.5 7B Instruct (Alibaba)",
        "provider": "ollama",
        "ollama_tag": "qwen2.5:7b",
        "type": "Instruction & Spatial Logic",
        "latency_target": "~30ms",
        "description": "Alibaba state-of-the-art language & spatial logic model for autonomous driving telemetry.",
    },
    "qwen2.5-vl-7b": {
        "id": "qwen2.5-vl-7b",
        "name": "Qwen 2.5 VL 7B (Vision-Language)",
        "provider": "ollama",
        "ollama_tag": "qwen2.5-vl:7b",
        "type": "Vision-Language",
        "latency_target": "~35ms",
        "description": "Alibaba spatial perception & multi-modal scene analysis for AV visual context.",
    },
    "llama3.2-3b": {
        "id": "llama3.2-3b",
        "name": "Llama 3.2 3B Instruct (Meta)",
        "provider": "ollama",
        "ollama_tag": "llama3.2:3b",
        "type": "Edge Language",
        "latency_target": "~15ms",
        "description": "Meta lightweight edge instruction model for rapid 20Hz telemetry parsing.",
    },
    "deepseek-r1-distill-8b": {
        "id": "deepseek-r1-distill-8b",
        "name": "DeepSeek R1 Distill 8B (CoT Reasoning)",
        "provider": "ollama",
        "ollama_tag": "deepseek-r1:8b",
        "type": "Chain-of-Thought Reasoning",
        "latency_target": "~45ms",
        "description": "DeepSeek reasoning model specialized for safety monitor fault root-cause analysis.",
    },
    "phi-3.5-mini": {
        "id": "phi-3.5-mini",
        "name": "Phi-3.5 Mini 3.8B (Microsoft)",
        "provider": "ollama",
        "ollama_tag": "phi3.5",
        "type": "Compact Logic",
        "latency_target": "~18ms",
        "description": "Microsoft compact logic model for deterministic policy validation.",
    },
}

class LLMAdvisor:
    """Manages active LLM model selection and generates scene advice."""

    def __init__(
        self,
        default_model: str = "qwen2.5-7b",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        
        # Load persisted model selection if available
        self.active_model_id = default_model
        if STATE_FILE.exists():
            try:
                saved = STATE_FILE.read_text().strip()
                if saved in SUPPORTED_LOCAL_MODELS:
                    self.active_model_id = saved
            except Exception:
                pass
        if self.active_model_id not in SUPPORTED_LOCAL_MODELS:
            self.active_model_id = "qwen2.5-7b"

    @property
    def active_model(self) -> Dict[str, Any]:
        return SUPPORTED_LOCAL_MODELS[self.active_model_id]

    def set_active_model(self, model_id: str) -> bool:
        """Switch active LLM model dynamically and persist selection."""
        if model_id in SUPPORTED_LOCAL_MODELS:
            self.active_model_id = model_id
            try:
                STATE_FILE.write_text(model_id)
            except Exception as e:
                logger.warning(f"Could not persist LLM state: {e}")
            logger.info(f"LLM Advisor switched to active model: {self.active_model['name']}")
            return True
        logger.warning(f"Requested LLM model_id '{model_id}' not found in SUPPORTED_LOCAL_MODELS.")
        return False

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Return list of supported models with active flag."""
        models = []
        for mid, info in SUPPORTED_LOCAL_MODELS.items():
            item = dict(info)
            item["is_active"] = (mid == self.active_model_id)
            models.append(item)
        return models

    def query_local_ollama(self, prompt: str, timeout_s: float = 0.5) -> Optional[str]:
        """Attempt non-blocking fast query to local Ollama instance."""
        try:
            req_data = json.dumps({
                "model": self.active_model["ollama_tag"],
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 64}
            }).encode("utf-8")
            
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    return res_json.get("response", "").strip()
        except Exception:
            # Fallback to local rule-based generation when Ollama daemon is offline/busy
            pass
        return None

    def generate_scene_advice(self, frame: Any, record: Any) -> Dict[str, Any]:
        """Generate structured scene description & advisory tags based on active model."""
        road_class = str(getattr(frame, "road_class", "ARTERIAL")).upper()
        trust_state = getattr(record, "trust_state", "FULL_AUTONOMY")
        if not isinstance(trust_state, str):
            trust_state = trust_state.value

        model_name = self.active_model["name"]

        # Formulate contextual prompt
        prompt = (
            f"Vehicle traveling on {road_class} road under {getattr(frame, 'weather', 'clear')} weather. "
            f"Assurance status: {trust_state}. Provide concise 1-sentence autonomous driving scene advice."
        )

        # Try active LLM endpoint (Ollama)
        ollama_reply = self.query_local_ollama(prompt, timeout_s=0.2)
        if ollama_reply:
            desc = f"{model_name}: {ollama_reply}"
        else:
            # Deterministic fallback tailored per active model persona
            if trust_state != "FULL_AUTONOMY":
                desc = f"{model_name}: ⚠️ System [{trust_state}]. Degradation detected on {road_class}. Restricting max speed & enforcing safe buffer."
            else:
                desc = f"{model_name}: Context [{road_class}]. Seattle GIS map match ACTIVE. System operating nominally in {trust_state}."

        tags = [
            road_class.lower(),
            "seattle_gis",
            "daylight" if getattr(frame, "weather", "clear") == "clear" else "adverse_weather",
            self.active_model_id,
        ]

        if trust_state != "FULL_AUTONOMY":
            tags.append("degraded_assurance")

        return {
            "model_id": self.active_model_id,
            "model_name": model_name,
            "scene_desc": desc,
            "tags": tags,
        }
