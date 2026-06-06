"""Vendored cross-repo ripple mesh (sibling: reactor-core).

ripple_contract.py is a BYTE-IDENTICAL vendored copy of JARVIS's portable
verification contract. Independent verification by design — reactor-core trusts
a JARVIS ripple ONLY after verifying its HMAC/nonce/TTL/origin locally, and
NEVER executes anything JARVIS sends ("predictions, not requests").
"""
