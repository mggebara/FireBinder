Planted-conflict fixtures for reconcile: one JSON file per case.
Each file is {"name", "description", "evidence": [Evidence...], "expected_conflicts": [{"field", "severity"}]}; Evidence matches firebinder.state.Evidence.
tests/test_reconcile.py asserts find_conflicts(evidence) yields exactly the expected (field, severity) set — no more, no fewer.
