"""MultiDatasetPipeline: Composite, multi-modal dataset pipeline.

Fuses primary sensor data (NVIDIA PhysicalAI / comma2k19 / Synthetic) with
spatial map-matching overlays (Seattle Open Data GIS) into a unified,
bounded-memory streaming `Iterator[HealthFrame]`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

from sentinel.pipeline.readers.base import Reader
from sentinel.pipeline.readers.synthetic import ScenarioSpec, SyntheticReader
from sentinel.pipeline.readers.seattle_gis import SeattleGISOverlay
from sentinel.schemas.health_frame import HealthFrame

logger = logging.getLogger(__name__)


import random

class MultiDatasetPipeline(Reader):
    """Composite pipeline that synchronizes multiple dataset sources.
    
    Automatically cycles through available scenarios & dataset clips across server runs.
    """

    source_name = "multi_dataset_pipeline"

    SCENARIO_PRESETS = [
        {"name": "nvidia_physicalai_clip001", "display": "NVIDIA PhysicalAI Clip 001 + Seattle GIS", "source": "nvidia_physicalai", "seed": 101, "speed": 18.5, "weather": "clear"},
        {"name": "comma2k19_highway_chunk1", "display": "comma2k19 Highway Chunk 1 + Seattle GIS", "source": "comma2k19", "seed": 202, "speed": 28.0, "weather": "clear"},
        {"name": "seattle_downtown_construction", "display": "Seattle Open Data GIS (Pike & Mercer Construction Zone)", "source": "seattle_gis", "seed": 303, "speed": 12.0, "weather": "rain"},
    ]

    def __init__(
        self,
        primary_source: str = "nvidia_physicalai",
        data_dir: Optional[Union[str, Path]] = None,
        gis_dir: Optional[Union[str, Path]] = None,
        scenario_spec: Optional[ScenarioSpec] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.primary_source = primary_source
        self.data_dir = Path(data_dir) if data_dir else Path("data_real")
        self.gis_dir = Path(gis_dir) if gis_dir else Path("sentinel/data_real/seattle")
        
        # Scenario selection: advance preset index on each new instantiation (e.g. server restart / browser reconnect)
        state_file = Path("/tmp/sentinel_dataset_switch.state")
        preset_idx = 0
        if state_file.exists():
            try:
                preset_idx = (int(state_file.read_text().strip()) + 1) % len(self.SCENARIO_PRESETS)
            except Exception:
                preset_idx = 0
        try:
            state_file.write_text(str(preset_idx))
        except Exception:
            pass

        self.preset_idx = preset_idx
        self.preset = self.SCENARIO_PRESETS[self.preset_idx]
        self.display_name = self.preset["display"]
        self.primary_source = self.preset.get("source", primary_source)
        self.seed = seed if seed is not None else self.preset["seed"]
        
        self.scenario_spec = scenario_spec or ScenarioSpec(
            duration_s=3600.0,
            nominal_speed_mps=self.preset["speed"],
            name=self.preset["name"]
        )

        # Initialize GIS overlay if available
        self.gis_overlay: Optional[SeattleGISOverlay] = None
        if self.gis_dir.exists():
            try:
                self.gis_overlay = SeattleGISOverlay(self.gis_dir)
                logger.info(f"Loaded Seattle GIS spatial overlay from {self.gis_dir}")
            except Exception as e:
                logger.warning(f"Failed to load Seattle GIS overlay: {e}")

        # Initialize primary reader with graceful fallback
        self.primary_reader: Reader = self._build_primary_reader()

    def _build_primary_reader(self) -> Reader:
        if self.primary_source == "nvidia_physicalai":
            try:
                from sentinel.pipeline.readers.physicalai import PhysicalAIReader
                return PhysicalAIReader(self.data_dir / "physical_ai" / "PhysicalAI_Sample_Clip_001")
            except Exception as e:
                logger.warning(f"PhysicalAI dataset unavailable ({e}), falling back to Synthetic generator")

        elif self.primary_source == "comma2k19":
            try:
                from sentinel.pipeline.readers.comma2k19 import Comma2k19Reader
                return Comma2k19Reader(self.data_dir / "comma2k19" / "Chunk_1")
            except Exception as e:
                logger.warning(f"comma2k19 dataset unavailable ({e}), falling back to Synthetic generator")

        # Default fallback
        return SyntheticReader(self.scenario_spec, seed=self.seed)

    def __iter__(self) -> Iterator[HealthFrame]:
        """Stream frames from primary reader, enriched by Seattle GIS map matching."""
        try:
            for frame in self.primary_reader:
                if self.gis_overlay is not None:
                    frame = self.gis_overlay.match_frame(frame)
                yield frame
        except NotImplementedError as e:
            logger.warning(f"Primary reader {self.primary_source} raised NotImplementedError ({e}). Falling back to SyntheticReader ({self.display_name}).")
            fallback = SyntheticReader(self.scenario_spec, seed=self.seed)
            for frame in fallback:
                if self.gis_overlay is not None:
                    frame = self.gis_overlay.match_frame(frame)
                yield frame
