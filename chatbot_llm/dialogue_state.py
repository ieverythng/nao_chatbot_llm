# Copyright (c) 2026 TODO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-dialogue state and the registry that holds the set of open dialogues."""

import threading
from dataclasses import dataclass, field
from typing import Iterator, List, Optional
from uuid import UUID


@dataclass
class Dialogue:
    """The in-memory state of a single open dialogue.

    Owned by a DialoguesRegistry. Mutations to `msgs_history` and
    `result` must be done while holding the registry's lock; reads
    that need a consistent snapshot must do the same.
    """

    id: UUID
    role: str
    role_configuration: str = "{}"
    msgs_history: List[dict] = field(default_factory=list)
    result: Optional[object] = None
    done: threading.Event = field(default_factory=threading.Event)


class DialoguesRegistry:
    """Thread-safe map of UUID -> Dialogue.

    The lock guards both the dict structure and the per-dialogue
    fields. It is intentionally a single lock for simplicity: callers
    are expected to hold it only briefly, never across LLM/network
    round-trips.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._lock = threading.Lock()
        self._dialogues: dict = {}

    @property
    def lock(self) -> threading.Lock:
        """The single mutex guarding the registry and all dialogues it holds."""
        return self._lock

    def add(self, dialogue: Dialogue) -> bool:
        """Insert `dialogue`; return False if a dialogue with the same id is already present."""
        with self._lock:
            if dialogue.id in self._dialogues:
                return False
            self._dialogues[dialogue.id] = dialogue
            return True

    def get(self, dialogue_id: UUID) -> Optional[Dialogue]:
        """Return the dialogue with id `dialogue_id`, or None."""
        with self._lock:
            return self._dialogues.get(dialogue_id)

    def remove(self, dialogue_id: UUID) -> Optional[Dialogue]:
        """Remove and return the dialogue with id `dialogue_id`, or None."""
        with self._lock:
            return self._dialogues.pop(dialogue_id, None)

    def __contains__(self, dialogue_id: UUID) -> bool:
        """Return True iff a dialogue with `dialogue_id` is currently open."""
        with self._lock:
            return dialogue_id in self._dialogues

    def __len__(self) -> int:
        """Return the number of currently-open dialogues."""
        with self._lock:
            return len(self._dialogues)

    def ids(self) -> List[UUID]:
        """Return a snapshot list of currently-open dialogue ids."""
        with self._lock:
            return list(self._dialogues.keys())

    def snapshot(self) -> Iterator[Dialogue]:
        """Iterate over a snapshot of the currently-open dialogues."""
        with self._lock:
            return iter(list(self._dialogues.values()))
