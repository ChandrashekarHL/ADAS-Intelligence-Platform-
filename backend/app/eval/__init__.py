"""M11 — evaluation and regression suite (spec FR-13, release gates §19).

A locked set of synthetic AEB scenarios with hand-pinned expectations is run through the
deterministic pipeline (quality gates → metrics → events) and, when a real provider is
configured, through the diagnostic agent and verifier. Every case is scored against its
expectations; the aggregates are checked against the release gate and, optionally,
against a stored baseline so a model, prompt or code change can never silently make the
agent worse.
"""
