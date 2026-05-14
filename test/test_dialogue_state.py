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

"""Unit tests for chatbot_llm.dialogue_state."""

import threading
from uuid import uuid4

from chatbot_llm.dialogue_state import Dialogue, DialoguesRegistry


def _make_dialogue(role: str = "__default__") -> Dialogue:
    return Dialogue(id=uuid4(), role=role)


class TestDialoguesRegistry:
    """Tests for DialoguesRegistry basic operations."""

    def test_add_returns_true_on_first_insert(self):
        """add() returns True when the id is new."""
        r = DialoguesRegistry()
        assert r.add(_make_dialogue()) is True

    def test_add_returns_false_on_duplicate_id(self):
        """add() returns False if a dialogue with the same id is already present."""
        r = DialoguesRegistry()
        d = _make_dialogue()
        assert r.add(d) is True
        # Build a second Dialogue with the same id; should be rejected.
        clone = Dialogue(id=d.id, role="__ask__")
        assert r.add(clone) is False

    def test_get_returns_inserted_dialogue(self):
        """get() returns the same object that was add()ed."""
        r = DialoguesRegistry()
        d = _make_dialogue()
        r.add(d)
        assert r.get(d.id) is d

    def test_get_unknown_returns_none(self):
        """get() of an unknown id returns None."""
        r = DialoguesRegistry()
        assert r.get(uuid4()) is None

    def test_remove_returns_dialogue_and_drops_it(self):
        """remove() returns the dialogue and the registry no longer contains it."""
        r = DialoguesRegistry()
        d = _make_dialogue()
        r.add(d)
        assert r.remove(d.id) is d
        assert d.id not in r
        assert r.get(d.id) is None

    def test_remove_unknown_returns_none(self):
        """remove() of an unknown id returns None and does not raise."""
        r = DialoguesRegistry()
        assert r.remove(uuid4()) is None

    def test_len_and_ids(self):
        """__len__ and ids() reflect the open dialogues."""
        r = DialoguesRegistry()
        assert len(r) == 0
        assert r.ids() == []

        d1 = _make_dialogue()
        d2 = _make_dialogue("__ask__")
        r.add(d1)
        r.add(d2)
        assert len(r) == 2
        assert set(r.ids()) == {d1.id, d2.id}

    def test_contains(self):
        """`uuid in registry` reflects presence."""
        r = DialoguesRegistry()
        d = _make_dialogue()
        assert d.id not in r
        r.add(d)
        assert d.id in r

    def test_snapshot_iterates_independently_of_subsequent_mutations(self):
        """snapshot() returns an iterator over a copy; later changes don't affect it."""
        r = DialoguesRegistry()
        d1 = _make_dialogue()
        r.add(d1)
        snap = list(r.snapshot())
        d2 = _make_dialogue()
        r.add(d2)
        # The snapshot was taken before d2 was added.
        assert snap == [d1]


class TestConcurrency:
    """Smoke tests that the registry's lock prevents tearing under contention."""

    def test_concurrent_add_remove(self):
        """N threads adding+removing concurrently leaves the registry consistent."""
        r = DialoguesRegistry()
        errors: list = []

        def worker():
            try:
                for _ in range(200):
                    d = _make_dialogue()
                    assert r.add(d) is True
                    assert r.get(d.id) is d
                    assert r.remove(d.id) is d
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert len(r) == 0

    def test_concurrent_adds_to_full_dictionary(self):
        """All inserts succeed when ids are unique; final count is the total."""
        r = DialoguesRegistry()
        per_thread = 50
        n_threads = 4

        def worker():
            for _ in range(per_thread):
                r.add(_make_dialogue())

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(r) == per_thread * n_threads
