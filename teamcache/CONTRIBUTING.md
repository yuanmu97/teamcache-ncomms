# Contributing

Keep changes focused on the supported SGLang usage path. Research-only plotting,
large traces, cluster addresses, model weights, and paper build files do not
belong in this repository.

Before opening a change:

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests
teamcache selftest
```

Engine compatibility changes should include a synthetic `common.py` fixture
covering every rewritten site. Never weaken the exact-anchor checks merely to
make a new SGLang version patch successfully: update the edit and test the new
source shape explicitly.
