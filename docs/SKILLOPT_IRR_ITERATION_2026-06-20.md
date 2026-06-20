# SkillOpt IRR Iteration, 20 June 2026

## Objective

Add a one-call Intent-Route-Response ablation that binds response wording,
planner-handoff intent, grounded evidence, and canonical skill arguments without
regressing the existing response-first, planner completion, planner dialogue,
execution-report, or report-result paths.

## Locked Inputs

- Target artifacts: `config/chat_prompt_pack.yaml`, IRR structural schema, and
  the isolated `atomic_irr` turn-engine path.
- Train set: grounded bring-to-recipient, missing recipient, contradictory
  dialogue execution promise, canonical subject recall, and one-call behavior.
- Holdout set: existing dialogue/knowledge/execution routing plus planner
  completion, planner dialogue, execution report, and report-result wording.
- Acceptance gate: all train cases pass, all focused holdouts remain green,
  existing public ROS/planner contracts remain unchanged, and IRR uses one LLM
  call.

## Baseline

| Case | Expected | Baseline | Pass |
|---|---|---|---|
| Existing hooks branch focused suite | Preserve current behavior | 57 passed | Yes |
| Atomic route/intent/response | One bound decision | No isolated contract | No |
| Canonical role-to-entity guard | Reject incomplete grounded execution | No pre-handoff registry guard | No |
| Subject-focused evidence | Resolve known canonical subject | Broad snapshot only | No |

## Mutation Batch

1. `add`: add one isolated IRR prompt block requiring atomic route, intent,
   response, handoff request, and evidence binding.
2. `replace`: represent canonical targets as role-bearing `intent.arguments`
   instead of a flat target-ID list.
3. `delete`: omit model-owned `speak_now` and authoritative `publish`; speech
   ownership and actual publication remain deterministic runtime decisions.

## Train Results

| Case | Before | After | Delta |
|---|---|---|---|
| Grounded object and recipient | Unsupported | Accepted with canonical bindings | Improved |
| Missing recipient | Could reach planner | One clarification, no handoff | Improved |
| Dialogue action promise | Route/text could disagree | Replaced by clarification | Improved |
| Canonical subject mention | Broad lookup only | Bounded subject lookup seam | Improved |
| Atomic model call | No isolated mode | One call using IRR schema | Improved |

The eight new IRR/turn-state/atomic-engine tests pass.

## Holdout Results

| Holdout | Result |
|---|---|
| Planner completion wording | Pass |
| Planner dialogue wording | Pass |
| Execution-report wording | Pass |
| Report-result wording | Pass |
| Dedicated completion/dialogue/report holdouts | 8 passed |
| Focused chatbot/planner-handoff suite | 85 passed |
| Broad import-light suite | 138 passed; one unrelated missing `interaction_skills` fixture |

## Decision

- Accept the mutation.
- No rollback is required.
- Keep `response_first` as the runtime default until the ROS-PC A/B sweep.
- Do not modify the existing system-turn wording prompts in this iteration.

## Next Mutation Hypothesis

After live evidence, tune only the smallest failed IRR invariant. The first
candidate is evidence selection for ambiguous aliases; do not add wording unless
the paired response-first/intent-first/atomic-IRR trace demonstrates the need.
