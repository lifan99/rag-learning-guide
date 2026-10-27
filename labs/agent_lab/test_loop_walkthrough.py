import unittest
from loop_walkthrough import walkthrough


class WalkthroughTests(unittest.TestCase):
    def test_observations_and_confirmation_boundary(self):
        result=walkthrough()
        self.assertEqual([t['action'] for t in result['turns']],
                         ['lookup_policy','lookup_employee','prepare_application'])
        self.assertTrue(all(t['observation'] for t in result['turns']))
        self.assertEqual((result['status'],result['applications']),('awaiting_confirmation',0))

    def test_step_limit_stops_without_creating_an_application(self):
        result=walkthrough(max_steps=1)
        self.assertEqual(len(result['turns']),1)
        self.assertEqual((result['status'],result['applications']),('budget_exhausted',0))

    def test_unlisted_tool_is_rejected_before_execution(self):
        result=walkthrough(invalid_tool=True)
        self.assertEqual(result['turns'],[])
        self.assertEqual((result['status'],result['applications']),('failed',0))


if __name__ == '__main__':
    unittest.main()
