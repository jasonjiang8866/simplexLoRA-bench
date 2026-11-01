"""Read-only diagnostics. Emit only fixed categories and selected resource fields."""
import json
import signal
import urllib.error
import urllib.request

from provider import API, REPOSITORY, NoRedirect, Runpod

PHASES = {'starting', 'global-volume-write-read', 'cuda-gate-and-six-smoke-workers',
          'publish-completed-results', 'fresh-pod-global-restore', 'complete'}


def main():
    api = Runpod()
    for pod in api.listing('/pods', 'pods'):
        if pod.get('env', {}).get('SIMPLEX_MANAGED_BY') != REPOSITORY:
            continue
        detail = api.call('/pods/' + pod['id'])
        print(json.dumps({k: detail.get(k) for k in
              ('id', 'name', 'status', 'dataCenterId', 'cost')}), flush=True)
        token = detail.get('env', {}).get('SIMPLEX_RESULT_TOKEN')
        if token:
            req = urllib.request.Request(f"https://{pod['id']}-8888.proxy.runpod.net/result",
                                         headers={'Authorization': 'Bearer ' + token})
            try:
                with urllib.request.build_opener(NoRedirect).open(req, timeout=10) as response:
                    result = json.loads(response.read(2_000_001))
                print(json.dumps({'worker_status': result.get('status') if result.get('status') in
                    {'running', 'passed', 'failed'} else 'unknown',
                    'worker_phase': result.get('phase') if result.get('phase') in PHASES else 'unknown'}), flush=True)
            except urllib.error.HTTPError as error:
                # Inspect in memory; neither the body nor arbitrary headers leave this process.
                body = error.read(16000).decode(errors='replace').lower()
                categories = [label for needle, label in (
                    ('cloudflare', 'cloudflare-response'), ('access denied', 'access-denied'),
                    ('forbidden', 'forbidden'), ('not ready', 'not-ready'),
                    ('authentication', 'authentication-mentioned'), ('unauthorized', 'unauthorized'),
                    ('not found', 'not-found'), ('proxy', 'proxy-mentioned')) if needle in body]
                print(json.dumps({'http_status': error.code, 'categories': categories}), flush=True)
            except Exception as error:
                print('Worker check failed: ' + type(error).__name__, flush=True)
        req = urllib.request.Request(API + '/pods/' + pod['id'] + '/logs?tail=30',
                                     headers={'Authorization': 'Bearer ' + api.key})
        def timeout(*_):
            raise TimeoutError()
        old = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(20)
        try:
            with urllib.request.build_opener(NoRedirect).open(req, timeout=10) as stream:
                for index, raw in enumerate(stream):
                    if raw.startswith(b'data:'):
                        event = json.loads(raw[5:])
                        line = str(event.get('line', '')).lower()
                        categories = [label for needle, label in (
                            ('pull complete', 'image-layer-pulled'), ('downloaded newer image', 'image-downloaded'),
                            ('start container', 'container-start'), ('extracting', 'extracting-layer'),
                            ('global store: mounting', 'global-volume-mount'), ('traceback', 'python-traceback'),
                            ('modulenotfounderror', 'missing-python-module'), ('permission denied', 'permission-denied'),
                            ('error', 'error-mentioned'), ('listening', 'listening')) if needle in line]
                        if categories:
                            print(json.dumps({'source': event.get('source') if event.get('source') in
                                {'system', 'container'} else 'unknown', 'categories': categories}), flush=True)
                    if index > 200:
                        break
        except Exception as error:
            print('Log read ended: ' + type(error).__name__, flush=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


if __name__ == '__main__':
    main()
