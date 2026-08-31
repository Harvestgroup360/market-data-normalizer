## What this changes

<!-- One or two sentences. -->

## Does it change any number the library already returns?

- [ ] No — same outputs, verified by the existing suite
- [ ] Yes — and the new value is correct, because:

<!--
If yes, say how you know. This is the question we care most about: a change
that alters a published number to save time is a different proposal from one
that makes the same number arrive sooner, and the two get judged differently.
-->

## Checks

- [ ] `pytest -q` passes
- [ ] New behaviour has a test that fails without the change
- [ ] No new runtime dependency
- [ ] No new default constant that cannot be right for every market

<!--
The house rules are in CONTRIBUTING.md. The one worth repeating: nothing looks
forward, and there is a property test that pins it.
-->
