"""Hardware profiler for NVIDIA Grace Blackwell GB10 & Local LLM Benchmarking."""

from __future__ import annotations
import time
import asyncio
import httpx
from typing import Dict, Any

class HardwareProfiler:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "nemotron-3.5-lightning"):
        self.ollama_url = ollama_url
        self.model = model
        self.metrics: Dict[str, Any] = {
            "decision_latencies_us": [],
            "llm_ttft_ms": [],
            "llm_tokens_per_sec": [],
            "total_llm_calls": 0,
            "failed_llm_calls": 0
        }

    def record_decision_latency(self, latency_us: float) -> None:
        self.metrics["decision_latencies_us"].append(latency_us)

    async def benchmark_llm_advisor(self, prompt: str = "Assess autonomous driving state context.") -> Dict[str, float]:
        start_time = time.perf_counter()
        ttft = 0.0
        tokens_count = 0
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False}
                )
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                if response.status_code == 200:
                    data = response.json()
                    eval_count = data.get("eval_count", 20)
                    eval_duration_ns = data.get("eval_duration", 1)
                    tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0
                    ttft = data.get("prompt_eval_duration", 0) / 1e6
                    self.metrics["llm_ttft_ms"].append(ttft)
                    self.metrics["llm_tokens_per_sec"].append(tps)
                    self.metrics["total_llm_calls"] += 1
                    return {"ttft_ms": ttft, "tps": tps, "total_ms": duration_ms}
                else:
                    self.metrics["failed_llm_calls"] += 1
        except Exception:
            self.metrics["failed_llm_calls"] += 1

        return {"ttft_ms": 0.0, "tps": 0.0, "total_ms": 0.0}

    def get_summary(self) -> Dict[str, Any]:
        latencies = self.metrics["decision_latencies_us"]
        p50 = sorted(latencies)[int(len(latencies) * 0.5)] if latencies else 0.0
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0
        
        ttfts = self.metrics["llm_ttft_ms"]
        tps_list = self.metrics["llm_tokens_per_sec"]

        return {
            "p50_us": p50,
            "p95_us": p95,
            "p99_us": p99,
            "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0.0,
            "avg_tps": sum(tps_list) / len(tps_list) if tps_list else 0.0,
            "total_llm_calls": self.metrics["total_llm_calls"],
            "failed_llm_calls": self.metrics["failed_llm_calls"]
        }
