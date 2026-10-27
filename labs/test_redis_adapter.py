import unittest
from redis_adapter import RedisCache
from cache_lab import Access, CacheUnavailable, Source, ask


class SimulatedRedisError(Exception):
    pass


class ClientDouble:
    def __init__(self):
        self.values = {}
        self.expiries = {}
        self.failed = False
    def get(self, key):
        if self.failed:
            raise SimulatedRedisError('offline')
        return self.values.get(key)
    def set(self, key, value, *, ex):
        self.values[key] = value
        self.expiries[key] = ex
    def delete(self, key):
        self.values.pop(key, None)


class AdapterContractTests(unittest.TestCase):
    def test_string_and_expiry_are_sent_together_then_deleted(self):
        client = ClientDouble(); cache = RedisCache(client,(SimulatedRedisError,))
        cache.set('own-test-key','{"answer":2000}',30)
        self.assertEqual(client.expiries['own-test-key'],30)
        self.assertEqual(cache.get('own-test-key'),'{"answer":2000}')
        cache.delete('own-test-key')
        self.assertIsNone(cache.get('own-test-key'))

    def test_connection_failure_falls_back_only_for_current_authorized_access(self):
        client = ClientDouble(); client.failed = True
        cache = RedisCache(client,(SimulatedRedisError,)); source = Source()
        self.assertEqual(ask('q',Access(),'v1',cache,source)['cache'],'bypass')
        with self.assertRaises(PermissionError):
            ask('q',Access(allowed=False),'v1',cache,source)
        self.assertEqual(source.calls,1)

    def test_non_positive_expiry_is_rejected(self):
        cache=RedisCache(ClientDouble(),(SimulatedRedisError,))
        for value in [0,-1,True,1.5]:
            with self.assertRaises(ValueError): cache.set('key','value',value)


if __name__ == '__main__':
    unittest.main()
