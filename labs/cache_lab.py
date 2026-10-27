"""Offline cache-aside exercise; no Redis server, network or model calls."""
from dataclasses import dataclass, replace
import hashlib
import json


class CacheUnavailable(Exception):
    pass


class Clock:
    def __init__(self):
        self.now = 0

    def advance(self, seconds):
        self.now += seconds


class MemoryTTLCache:
    """A controllable dictionary substitute, not Redis or its eviction model."""
    def __init__(self, clock):
        self.clock = clock
        self.entries = {}
        self.available = True

    def check(self):
        if not self.available:
            raise CacheUnavailable('simulated cache outage')

    def get(self, key):
        self.check()
        entry = self.entries.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if self.clock.now >= expires_at:
            del self.entries[key]
            return None
        return value

    def set(self, key, value, ttl):
        self.check()
        if ttl <= 0:
            raise ValueError('ttl must be positive')
        self.entries[key] = (value, self.clock.now + ttl)

    def delete(self, key):
        self.check()
        self.entries.pop(key, None)


@dataclass(frozen=True)
class Access:
    # In a service these values come from current server-side authorization.
    user: str = 'employee001'
    tenant: str = 'demo-company'
    groups: tuple = ('employee',)
    permission_version: str = 'acl-v1'
    allowed: bool = True


def cache_key(question, access, kb_version, config_version):
    identity = [access.tenant, access.user, sorted(access.groups),
                access.permission_version, kb_version, config_version, question]
    raw = json.dumps(identity, ensure_ascii=False, separators=(',', ':'))
    return 'rag-cache-lab:' + hashlib.sha256(raw.encode()).hexdigest()


class Source:
    def __init__(self):
        self.calls = 0
        self.answer = '年度补贴上限为2000元。'

    def read(self, question):
        self.calls += 1
        return {'answer': self.answer, 'question': question}


def ask(question, access, kb_version, cache, source, ttl=30,
        config_version='model-prompt-retrieval-v1'):
    # Check current permission even for a warm cache; never bypass on outage.
    if not access.allowed:
        raise PermissionError('current identity cannot read this resource')
    key = cache_key(question, access, kb_version, config_version)
    cache_ok = True
    try:
        value = cache.get(key)
    except CacheUnavailable:
        value, cache_ok = None, False
    if value is not None:
        return {'cache': 'hit', 'result': json.loads(value), 'key': key}
    result = source.read(question)
    if cache_ok:
        try:
            cache.set(key, json.dumps(result, ensure_ascii=False), ttl)
        except CacheUnavailable:
            cache_ok = False
    return {'cache': 'miss' if cache_ok else 'bypass', 'result': result, 'key': key}


def demo():
    clock = Clock()
    cache, source, access = MemoryTTLCache(clock), Source(), Access()
    question = '培训补贴上限是多少？'
    rows = []

    def record(label, version='kb-v1', who=access):
        result = ask(question, who, version, cache, source)
        rows.append({'step': label, 'cache': result['cache'],
                     'source_calls': source.calls, 'answer': result['result']['answer']})
        return result

    first = record('first')
    record('repeat')
    clock.advance(31)
    record('expired')
    # Commit the authoritative update first, then invalidate the old entry.
    source.answer = '教学更新后的年度上限为2500元。'
    cache.delete(first['key'])
    record('updated', version='kb-v2')
    record('different_user', version='kb-v2', who=replace(access, user='employee002'))
    try:
        ask(question, replace(access, allowed=False), 'kb-v2', cache, source)
    except PermissionError:
        rows.append({'step': 'revoked', 'cache': 'denied', 'source_calls': source.calls})
    cache.available = False
    record('cache_unavailable', version='kb-v2')
    return {'mode': 'memory_mock', 'rows': rows,
            'note': 'Counts measure deterministic source calls, not Redis latency or LLM quality.'}


if __name__ == '__main__':
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
