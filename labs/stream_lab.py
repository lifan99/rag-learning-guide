"""A Flask SSE contract exercise; fixture checks are not semantic evaluation."""
import argparse
import json


FAKE_EVIDENCE = {'C1': '年度上限2000元。'}


def frame(event, data):
    return f'event: {event}\ndata: {json.dumps(data,ensure_ascii=False)}\n\n'


def fake_tokens(fail=False):
    yield '年度上限'
    if fail:
        raise TimeoutError('injected model timeout')
    yield '2000元。'


def fake_finalize(text):
    """Build a candidate for the one fixed fake evidence fixture."""
    return {'status': 'answered', 'answer': text, 'citations': ['C1']}


def validate_fake_final(result):
    """Check the envelope and fixed fixture, not real-world answer quality."""
    if not isinstance(result, dict) or set(result) != {'status', 'answer', 'citations'}:
        raise ValueError('invalid_final_fields')
    if result['status'] != 'answered' or not isinstance(result['answer'], str) or not result['answer'].strip():
        raise ValueError('invalid_final_status_or_text')
    citations = result['citations']
    if (not isinstance(citations, list) or not citations or
            not all(isinstance(cid, str) and cid in FAKE_EVIDENCE for cid in citations) or
            len(citations) != len(set(citations))):
        raise ValueError('invalid_final_citations')
    # Exact equality is deliberately limited to this deterministic fake fixture.
    # A live answer needs separate evidence/claim evaluation, not this assertion.
    if result['answer'] != '\n'.join(FAKE_EVIDENCE[cid] for cid in citations):
        raise ValueError('fake_fixture_mismatch')
    return result


def events(upstream, finalizer=fake_finalize, validator=validate_fake_final):
    pieces=[]
    try:
        try:
            yield frame('progress', {'stage':'generation','mode':'fake'})
            for piece in upstream:
                if not isinstance(piece, str):
                    raise ValueError('invalid_delta_text')
                pieces.append(piece)
                yield frame('delta', {'text':piece,'provisional':True})
            result = validator(finalizer(''.join(pieces)))
        finally:
            close=getattr(upstream,'close',None)
            if close:
                close()
    except Exception as exc:
        yield frame('error', {'error_type':type(exc).__name__,'complete':False})
    else:
        yield frame('final', {**result, 'mode':'fake'})


def create_app(provider=fake_tokens, finalizer=fake_finalize, validator=validate_fake_final):
    from flask import Flask, Response, jsonify, request
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=16384

    @app.post('/demo/stream')
    def stream():
        data=request.get_json(silent=True)
        if (not isinstance(data,dict) or set(data)!={'question'} or
                not isinstance(data['question'],str) or not data['question'].strip()
                or len(data['question'])>2000):
            return jsonify(error='expected_nonempty_question'),400
        # Demo-only provider; this endpoint has no real model or authentication.
        return Response(events(iter(provider()), finalizer, validator),mimetype='text/event-stream',
                        headers={'Cache-Control':'no-cache'})
    return app


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fail',action='store_true')
    args=parser.parse_args()
    with create_app(lambda:fake_tokens(args.fail)).test_client() as client:
        response=client.post('/demo/stream',json={'question':'培训补贴上限？'})
        print('HTTP',response.status_code,response.mimetype)
        print(response.get_data(as_text=True))
