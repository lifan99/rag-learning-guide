"""Pure node rules are tested independently; real graph test is opt-in."""
import importlib.util
import unittest
from langgraph_read_loop import initial_state, decide, execute_read, next_node, build_graph


def run_nodes(state):
    # This is a test harness, not a substitute marketed as LangGraph execution.
    for _ in range(4):
        state.update(decide(state))
        if next_node(state)=='stop':
            return state
        state.update(execute_read(state))
    raise AssertionError('Node rules failed to stop')


class NodeRulesTests(unittest.TestCase):
    def test_results_are_returned_to_state_before_completion(self):
        result=run_nodes(initial_state())
        self.assertEqual(result['status'],'completed')
        self.assertEqual([r['action'] for r in result['observations']],
                         ['lookup_policy','lookup_employee'])
        self.assertEqual(result['observations'][1]['observation']['months'],14)

    def test_step_budget_and_unknown_tool_have_different_outcomes(self):
        budget=run_nodes(initial_state(max_steps=1))
        self.assertEqual((budget['status'],len(budget['observations'])),('budget_exhausted',1))
        invalid=run_nodes(initial_state(invalid_tool=True))
        self.assertEqual((invalid['status'],invalid['observations']),('failed',[]))

    @unittest.skipUnless(importlib.util.find_spec('langgraph'), 'LangGraph not installed: node tests are not graph integration')
    def test_actual_graph_matches_three_node_rule_scenarios(self):
        graph=build_graph()
        for kwargs in [{},{'max_steps':1},{'invalid_tool':True}]:
            actual=graph.invoke(initial_state(**kwargs),{'recursion_limit':12})
            expected=run_nodes(initial_state(**kwargs))
            self.assertEqual(actual,expected)


if __name__ == '__main__':
    unittest.main()
