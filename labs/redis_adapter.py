"""Opt-in redis-py adapter. Importing this file does not connect to Redis."""
import argparse
import json
import os
import time
import uuid
from dataclasses import replace
from cache_lab import Access, CacheUnavailable, Source, ask


class RedisCache:
    def __init__(self, client, errors):
        self.client, self.errors = client, errors

    def get(self, key):
        try:
            return self.client.get(key)
        except self.errors as exc:
            raise CacheUnavailable('Redis read unavailable') from exc

    def set(self, key, value, ttl):
        if type(ttl) is not int or ttl <= 0:
            raise ValueError('ttl must be a positive integer')
        try:
            self.client.set(key, value, ex=ttl)
        except self.errors as exc:
            raise CacheUnavailable('Redis write unavailable') from exc

    def delete(self, key):
        try:
            self.client.delete(key)
        except self.errors as exc:
            raise CacheUnavailable('Redis deletion unavailable') from exc


def integration():
    import redis  # Install only in the learner's chosen isolated environment.
    url = os.environ.get('REDIS_URL')
    if not url:
        raise SystemExit('Set REDIS_URL for a dedicated learning instance; no connection made.')
    client = redis.Redis.from_url(url, decode_responses=True,
                                 socket_connect_timeout=1, socket_timeout=1)
    cache = RedisCache(client, (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError))
    source = Source()
    access = replace(Access(), tenant='course-probe-' + uuid.uuid4().hex)
    question = '培训补贴上限是多少？'
    touched, rows = set(), []

    def record(label, version='kb-v1'):
        result = ask(question, access, version, cache, source, ttl=2)
        touched.add(result['key'])
        rows.append({'step':label, 'cache':result['cache'],
                     'source_calls':source.calls, 'ttl':client.ttl(result['key']),
                     'answer':result['result']['answer']})
        return result

    try:
        client.ping()  # Fail explicitly if the selected instance is unavailable.
        first = record('first')
        record('repeat')
        time.sleep(2.2)
        record('expired')
        source.answer = '教学更新后的年度上限为2500元。'
        cache.delete(first['key'])
        deleted_ttl = client.ttl(first['key'])
        record('updated', 'kb-v2')
        assert [r['cache'] for r in rows] == ['miss','hit','miss','miss']
        assert [r['source_calls'] for r in rows] == [1,1,2,3]
        assert deleted_ttl == -2
        print(json.dumps({'mode':'real_redis', 'redis_py_version':redis.__version__,
                          'rows':rows, 'deleted_key_ttl':deleted_ttl},
                         ensure_ascii=False, indent=2))
    finally:
        # Delete only this probe's uniquely scoped keys. Never FLUSHDB/FLUSHALL.
        for key in touched:
            try:
                cache.delete(key)
            except CacheUnavailable:
                pass  # Probe keys already have a two-second TTL.
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true',
                        help='Connect to REDIS_URL and write temporary, namespaced keys.')
    args = parser.parse_args()
    if not args.run:
        parser.error('Use --run only after choosing a dedicated learning Redis instance.')
    integration()
