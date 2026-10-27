"""Small real-LangGraph scaffold; fixed decisions, read-only local tools.

Node functions can be tested without LangGraph. Running this CLI requires it.
No LLM, checkpoint, network tool or business write is provided here.
"""
import argparse
import json
from typing import TypedDict
from agent_lab import POLICY, employee_facts, validate_call


class State(TypedDict):
    owner: str
    plan: list[str]
    steps: int
    max_steps: int
    action: str
    observations: list[dict]
    status: str


def initial_state(max_steps=2, invalid_tool=False):
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError('max_steps must be a nonnegative integer')
    # The host supplies identity. Neither a model nor tool arguments choose it.
    return State(owner='employee001',
                 plan=['delete_all' if invalid_tool else 'lookup_policy','lookup_employee'],
                 steps=0, max_steps=max_steps, action='', observations=[], status='running')


def decide(state):
    if state['status'] == 'failed':
        return {'action':'', 'status':'failed'}
    if state['steps'] == len(state['plan']):
        return {'action':'', 'status':'completed'}
    if state['steps'] >= state['max_steps']:
        return {'action':'', 'status':'budget_exhausted'}
    # Replace only this controlled planner when adding actual model decisions.
    return {'action':state['plan'][state['steps']], 'status':'running'}


def execute_read(state):
    name = state['action']
    if name not in {'lookup_policy','lookup_employee'}:
        return {'status':'failed', 'action':''}
    validate_call(name,{})
    result = POLICY.copy() if name == 'lookup_policy' else employee_facts(state['owner'])
    observation = {'action':name, 'arguments':{}, 'observation':result}
    return {'steps':state['steps']+1,
            'observations':[*state['observations'],observation]}


def next_node(state):
    return 'execute' if state['status']=='running' else 'stop'


def build_graph():
    from langgraph.graph import StateGraph, START, END
    graph = StateGraph(State)
    graph.add_node('decide',decide)
    graph.add_node('execute',execute_read)
    graph.add_edge(START,'decide')
    graph.add_conditional_edges('decide',next_node,{'execute':'execute','stop':END})
    graph.add_edge('execute','decide')
    return graph.compile()  # No checkpointer: nothing survives a process restart.


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-steps',type=int,default=2)
    parser.add_argument('--invalid-tool',action='store_true')
    args=parser.parse_args()
    try:
        graph=build_graph()
    except ModuleNotFoundError as exc:
        if exc.name == 'langgraph':
            raise SystemExit('LangGraph is not installed in this interpreter; see the course framework exercise.') from None
        raise
    result=graph.invoke(initial_state(args.max_steps,args.invalid_tool),{'recursion_limit':12})
    print(json.dumps({'runtime':'langgraph','decision':'scripted','state':result},ensure_ascii=False,indent=2))
