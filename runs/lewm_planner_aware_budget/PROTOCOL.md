# Fixed PushT CEM qualification protocol

Status: frozen before reading any new smoke, tuning, or qualification outcome.

The exact machine-readable contract is `config.json` under
`planner_qualification`. The released `pusht/lewm` object checkpoint and local
`pusht_expert_train.lance` dataset are reused in place. Evaluation starts from a
stored dataset `state`, targets the state 25 rows later, and uses the installed
PushT `_set_state` and `_set_goal_state` reset path.

Task success is the environment termination rule: the joint agent/block
position error is below 20 and wrapped block-angle error is below `pi/9` within
50 environment steps. The dense task cost is cumulative Euclidean full-state
distance; normalized return is `1 - cost / (initial_distance * 50)`. These are
reported with final distance and per-start end-to-end latency.

Two deterministic smoke starts, six tuning starts, and twenty held-out starts
are sampled without replacement from valid dataset rows using three distinct
frozen NumPy seeds. Starts are independent evaluation units but may share a
source expert episode; episode IDs and row IDs are therefore reported so this
limitation is visible. There is no outcome-based replacement. Tuning and
held-out starts are disjoint by row.

The cheap smoke uses population 64, 3 CEM iterations, 8 elites, horizon 5, and
receding horizon 5. Only after it is mechanically valid may the five-point
bounded grid in `config.json` run. Candidate CEM randomness uses the same fixed
seed in a fresh process for every arm. The winner is selected by tuning success,
then mean task cost, median latency, and the declared lexicographic tie-break.
Its settings are locked before the held-out cohort is evaluated.

A competent baseline requires at least 50% success and at least 3 successes on
20 held-out starts, with mechanically valid finite metrics. Later regret-label
informativeness is reported separately: binary outcome saturation is allowed,
but continuous held-out task cost must have nonzero standard deviation and at
least three distinct values at `1e-6` precision. No adaptive controller,
refiner training, or fresh expensive cohort is authorized by this protocol.
