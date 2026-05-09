"""
dev/vulture_allowlist.py — Known false positives for the Layer 4.2 gate.

Vulture's whitelist mechanism: it parses this file like any other and
counts bare-name references as "uses". Items below are not really used
in this file — they exist only so vulture treats them as live and
suppresses the false positive elsewhere.

Run:
    vulture backend/ dev/vulture_allowlist.py --min-confidence 90

Add an entry here ONLY after verifying the flagged item is actually
referenced through indirection vulture can't see (kwargs forwarding,
chained assignment, runtime dispatch). Each entry comes with a comment
explaining why removing it would actually break something.
"""

# input_sanitizer.py:52 — ``ScanDecision`` is imported inside a try block
# and re-bound on ImportError via the chained assignment
# ``LlamaFirewall = ScannerType = UserMessage = Role = ScanDecision = _Stub``.
# Vulture sees the import only as a definition, not as the LHS of the
# chained assignment that re-binds it.
ScanDecision

# docker_manager.py:329 — ``_build_detail`` keeps ``wsl`` in its signature
# even though the current body branches only on the other flags. Callers
# pass ``wsl=...`` positionally for clarity; renaming silently breaks them.
wsl

# semantic_search.py:94 — ``init_vector_store`` accepts ``vector_dir`` and
# ``shared_model`` as forward-compat parameters: core/paths and the
# bundled-server bootstrap will override them once the migration lands.
vector_dir
shared_model
