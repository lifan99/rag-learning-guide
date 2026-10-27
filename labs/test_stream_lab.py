import json
import unittest
from stream_lab import create_app, events, fake_tokens, validate_fake_final


def parse_events(body):
    return [(lines[0].removeprefix('event: '),
             json.loads(lines[1].removeprefix('data: ')))
            for block in body.strip().split('\n\n')
            if (lines := block.splitlines())]


class TrackedTokens:
    def __init__(self, pieces, fail=False):
        self.pieces = iter(pieces)
        self.fail = fail
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.pieces)
        except StopIteration:
            if self.fail:
                raise TimeoutError('injected model timeout') from None
            raise

    def close(self):
        self.closed = True


class StreamTests(unittest.TestCase):
    def test_success_has_one_validated_final_after_deltas(self):
        source = TrackedTokens(fake_tokens())
        calls = []

        def finalizer(text):
            calls.append(('finalize', text))
            return {'status': 'answered', 'answer': text, 'citations': ['C1']}

        def validator(candidate):
            calls.append(('validate', candidate['answer']))
            return validate_fake_final(candidate)

        with create_app(lambda: source, finalizer, validator).test_client() as client:
            r = client.post('/demo/stream', json={'question': 'q'})
            frames = parse_events(r.get_data(as_text=True))
        self.assertEqual(r.mimetype, 'text/event-stream')
        self.assertEqual([event for event, _ in frames], ['progress', 'delta', 'delta', 'final'])
        text = ''.join(data['text'] for event, data in frames if event == 'delta')
        self.assertEqual(calls, [('finalize', text), ('validate', text)])
        self.assertTrue(all(data['provisional'] for event, data in frames if event == 'delta'))
        self.assertEqual(frames[-1][1], {'status': 'answered', 'answer': text,
                                      'citations': ['C1'], 'mode': 'fake'})
        self.assertTrue(source.closed)

    def assert_terminal_error(self, source, **kwargs):
        stream = events(source, **kwargs)
        frames = []
        for raw in stream:
            frame = parse_events(raw)[0]
            frames.append(frame)
            if frame[0] == 'error':
                # Cleanup precedes the terminal event; consumers can stop here.
                self.assertTrue(source.closed)
        names = [event for event, _ in frames]
        self.assertEqual(names[-1], 'error')
        self.assertEqual(names.count('error'), 1)
        self.assertNotIn('final', names)
        self.assertFalse(frames[-1][1]['complete'])

    def test_invalid_candidate_is_rejected_and_source_closed(self):
        valid_text = ''.join(fake_tokens())
        for candidate in (
            {'status': 'answered', 'answer': valid_text, 'citations': ['unknown']},
            {'status': 'answered', 'answer': valid_text, 'citations': []},
            {'status': 'answered', 'answer': valid_text, 'citations': ['C1', 'C1']},
            {'status': 'answered', 'answer': valid_text},
            {'status': 'invalid', 'answer': valid_text, 'citations': ['C1']},
            {'status': 'answered', 'answer': [], 'citations': ['C1']},
        ):
            with self.subTest(candidate=candidate):
                source = TrackedTokens(fake_tokens())
                self.assert_terminal_error(source, finalizer=lambda _, value=candidate: value)

    def test_empty_or_wrong_fake_text_never_becomes_final(self):
        for pieces in ([], [' '], ['wrong fake answer']):
            with self.subTest(pieces=pieces):
                self.assert_terminal_error(TrackedTokens(pieces))

    def test_finalizer_and_validator_failures_close_source(self):
        def fail(_):
            raise ValueError('private provider details must not be sent')

        for hook in ('finalizer', 'validator'):
            with self.subTest(hook=hook):
                source = TrackedTokens(fake_tokens())
                self.assert_terminal_error(source, **{hook: fail})

    def test_post_start_failure_is_an_error_event_without_final(self):
        source = TrackedTokens(['partial'], fail=True)
        with create_app(lambda: source).test_client() as client:
            r = client.post('/demo/stream', json={'question': 'q'})
            body = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        frames = parse_events(body)
        self.assertEqual([event for event, _ in frames], ['progress', 'delta', 'error'])
        self.assertEqual(frames[-1][1], {'error_type': 'TimeoutError', 'complete': False})
        self.assertNotIn('injected model timeout', body)
        self.assertTrue(source.closed)

    def test_bad_input_is_rejected_before_streaming(self):
        def should_not_start():
            self.fail('invalid input must not call the provider')

        with create_app(should_not_start).test_client() as client:
            r = client.post('/demo/stream', json={'question': []})
        self.assertEqual(r.status_code, 400)

    def test_disconnect_closes_the_upstream_iterator(self):
        source = TrackedTokens(['partial', 'unfinished'])
        stream = events(source)
        next(stream)
        next(stream)
        stream.close()
        self.assertTrue(source.closed)


if __name__ == '__main__':
    unittest.main()
