from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SessionMetadata:
    id: str
    created_at: str
    seed_instruction: str
    voice: str
    providers: dict[str, str]
    ended_cleanly: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

