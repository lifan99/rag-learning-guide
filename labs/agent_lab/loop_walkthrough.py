"""Observe the existing agent's action/result loop, without paid model calls."""
import argparse
import json
from pathlib import Path
import tempfile
from agent_lab import ScriptedDriver, Store, run_loop


class InvalidToolDriver(ScriptedDriver):
    def decide(self, messages, state, timeout):
        message, usage = super().decide(messages, state, timeout)
        message['tool_calls'][0]['function']['name'] = 'delete_all'
        return message, usage


def walkthrough(max_steps=6, invalid_tool=False):
    with tempfile.TemporaryDirectory() as folder:
        store = Store(Path(folder)/'loop.sqlite3')
        try:
            store.create('walkthrough')
            driver = InvalidToolDriver() if invalid_tool else ScriptedDriver()
            state = run_loop(store, 'walkthrough', 'employee001', driver,
                             max_steps=max_steps, max_seconds=30, context_bytes=12000)
            turns = []
            for decision, observation in state['history']:
                call = decision['tool_calls'][0]
                turns.append({'action': call['function']['name'],
                              'arguments': json.loads(call['function']['arguments']),
                              'observation': json.loads(observation['content'])})
            return {'mode':'scripted', 'turns':turns, 'status':state['status'],
                    'error':state.get('error'),
                    'applications':store.db.execute('SELECT COUNT(*) FROM applications').fetchone()[0]}
        finally:
            store.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--invalid-tool', action='store_true')
    args=parser.parse_args()
    print(json.dumps(walkthrough(args.max_steps,args.invalid_tool),ensure_ascii=False,indent=2))
