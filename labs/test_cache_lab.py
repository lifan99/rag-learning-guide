import unittest
from dataclasses import replace
from cache_lab import Access, Clock, MemoryTTLCache, Source, ask, cache_key


class CacheLabTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.cache = MemoryTTLCache(self.clock)
        self.source = Source()
        self.access = Access()

    def query(self, access=None, version='v1'):
        return ask('补贴上限？', access or self.access, version, self.cache, self.source)

    def test_repeat_avoids_source_until_expiry(self):
        self.assertEqual(self.query()['cache'], 'miss')
        self.assertEqual(self.query()['cache'], 'hit')
        self.assertEqual(self.source.calls, 1)
        self.clock.advance(30)
        self.assertEqual(self.query()['cache'], 'miss')
        self.assertEqual(self.source.calls, 2)

    def test_update_invalidation_and_version_do_not_return_old_answer(self):
        first = self.query()
        self.source.answer = '新版答案'
        self.cache.delete(first['key'])
        self.assertEqual(self.query()['result']['answer'], '新版答案')
        self.assertEqual(self.query(version='v2')['cache'], 'miss')

    def test_cache_identity_covers_scope_versions_and_configuration(self):
        base = cache_key('q', self.access, 'v1', 'config1')
        variants = [replace(self.access, user='other'),
                    replace(self.access, tenant='other'),
                    replace(self.access, groups=('finance',)),
                    replace(self.access, permission_version='acl-v2')]
        for access in variants:
            self.assertNotEqual(base, cache_key('q', access, 'v1', 'config1'))
        self.assertNotEqual(base, cache_key('q', self.access, 'v2', 'config1'))
        self.assertNotEqual(base, cache_key('q', self.access, 'v1', 'config2'))
        self.assertNotEqual(base, cache_key('different', self.access, 'v1', 'config1'))

    def test_other_user_misses_and_revoked_user_is_denied(self):
        self.query()
        self.assertEqual(self.query(replace(self.access, user='other'))['cache'], 'miss')
        before = self.source.calls
        with self.assertRaises(PermissionError):
            self.query(replace(self.access, allowed=False))
        self.assertEqual(self.source.calls, before)

    def test_cache_outage_falls_back_without_bypassing_authorization(self):
        self.cache.available = False
        self.assertEqual(self.query()['cache'], 'bypass')
        with self.assertRaises(PermissionError):
            self.query(replace(self.access, allowed=False))
        self.assertEqual(self.source.calls, 1)


if __name__ == '__main__':
    unittest.main()
