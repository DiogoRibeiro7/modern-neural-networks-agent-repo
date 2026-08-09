# Agent Handoff

Paste the following instruction into a coding agent after giving it access to the repository:

> Read `AGENTS.md` and `prompts/00_master_agent_prompt.md` in full. Inspect `STATUS.md`. Work on only the next atomic milestone unless blocked by shared infrastructure. For the selected track, verify the current primary paper and official repository before coding. Update the source registry, write the mathematical design note, add failing invariant tests, then implement the smallest correct mechanism. Run the full quality gate before stopping. Update `STATUS.md` with exact work completed, tests run, failures, and the next atomic milestone. Do not claim paper reproduction or SOTA performance unless `docs/claim_policy.md` permits it.

For the initial run, the selected track should be KAN unless you deliberately choose a different order and document why.
