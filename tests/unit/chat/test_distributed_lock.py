from agentic_rag.chat.distributed_lock import RedisMemoryLockManager


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    def set(self, key, value, nx=False, px=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        self.expirations[key] = px
        return True

    def eval(self, script, key_count, key, token, *args):
        if self.values.get(key) != token:
            return 0
        if "PEXPIRE" in script:
            self.expirations[key] = args[0]
            return 1
        self.values.pop(key, None)
        self.expirations.pop(key, None)
        return 1


def _manager(client):
    return RedisMemoryLockManager(
        client,
        wait_seconds=0,
        lease_seconds=10,
        renew_interval_seconds=60,
    )


def test_same_memory_id_is_exclusive_and_different_ids_are_parallel():
    client = FakeRedis()
    manager = _manager(client)

    first = manager.acquire("session-a")
    second = manager.acquire("session-a")
    other = manager.acquire("session-b")

    assert first is not None
    assert second is None
    assert other is not None
    first.release()
    other.release()


def test_renew_and_release_require_the_owner_token():
    client = FakeRedis()
    manager = _manager(client)
    lease = manager.acquire("session-a")
    assert lease is not None
    key = "agentic-rag:memory:session-a:lock"

    assert lease._renew_once()
    assert client.expirations[key] == 10_000
    client.values[key] = "new-owner-token"
    lease.release()

    assert client.values[key] == "new-owner-token"
