from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class SimulatorMeta(BaseModel):
    name: str
    description: str
    version: str = "0.1.0"
    tags: list[str] = Field(default_factory=list)


class SimulatorConfig(BaseModel):
    meta: SimulatorMeta


class Simulator(ABC):
    config: SimulatorConfig

    @abstractmethod
    def forward(self, X: Any):
        raise NotImplementedError
