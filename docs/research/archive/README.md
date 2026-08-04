# Archive — superseded plans and closed sub-studies

Nothing in this folder is current. It is kept for two reasons: writing up the study later needs
the *pre-registered* intent (what we predicted before seeing data), and several decisions only
make sense if you can see what they replaced.

**If you are picking up the project, read [`../README.md`](../README.md) instead.** It lists the
live docs and the current priority order.

Archived 2026-08-03.

| Doc | What it was | Why it is here |
|---|---|---|
| [`human-panel-pilot-plan.md`](./human-panel-pilot-plan.md) | Design + decision gates for buying human labels, drafted 2026-07-14 | **Executed and superseded.** Three runs completed 2026-07-29 → 08-01. Valuable as the honest pre-registration record: it says what we expected before any votes existed, including the close-pair gate that turned out to be wrong. Current policy is `../panel-study-playbook.md` |
| [`panel-pilot-execution-plan.md`](./panel-pilot-execution-plan.md) | "What do we actually do, in order" for the panel pilot | **Superseded 2026-07-31.** Carries its own ⚠️ marker: the close-pair "fail" it recorded was a bad gate, not a bad result. Ops procedure now lives in `../panel-pilot-runbook.md` |
| [`vlm-pilot-spec.md`](./vlm-pilot-spec.md) | Phase 0 spec — can a VLM pick the more attractive face, at what cost, which model | **Closed 2026-06, passed.** ~76–86% on 40 human-labeled pairs at ~$0.0008–0.004/pair, which authorised the 52k batch. Historical only, *except* §on the export API, which is still the reference for the `GET …/export` shape |
| [`pilot-500-results-brief.md`](./pilot-500-results-brief.md) | Dit + Alex each labelling the same 500 pairs vs Gemini, 2026-07-23 | **Superseded by the Prolific runs.** Its core finding — two in-house annotators cannot answer "do ordinary people agree" — is exactly why we bought a real panel. Numbers live on in log §2 and §5.5 |
| [`rating-app-build-plan.md`](./rating-app-build-plan.md) | Stack, data flow and build steps for the standalone rating web app | **Built, deployed, has carried four studies.** The app is `faceiq-rating`; operating it is `../panel-pilot-runbook.md` and `../prolific-soft-launch-form.md` |
| [`dev-team-brief.md`](./dev-team-brief.md) | One-page framing for the dev team, 2026-07-05 | **Stale.** Predates the BT refits, the panel work and every finding in log §5. Do not hand this to anyone as an introduction |

## Also moved out of `docs/research/`

[`../../ops/aws-gpu-training-setup.md`](../../ops/aws-gpu-training-setup.md) — renting an EC2
g5.xlarge, syncing code and photos, training, copying results back. **Still current and still
used** (runs v12–v15 went through it), it is just operations rather than research, so it now sits
under `docs/ops/` rather than cluttering the research folder. See also
`scripts/sync_panel_to_gpu.sh`, which automates most of it.
