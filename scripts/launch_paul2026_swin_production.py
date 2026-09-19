"""Submit production to an already deployed app and exit without waiting.

Deploy code (does not start training):
    modal deploy scripts/modal_paul2026_swin.py
Submit once, without --detach:
    modal run scripts/launch_paul2026_swin_production.py --system shared_hard --seed 42
Inspect later, independently of the launcher:
    modal app logs paul2026-swin --timestamps
    modal app logs paul2026-swin --function-call <fc-id> --timestamps

Save the printed FunctionCall ID for later inspection. The deployed function
retains the existing run directory and checkpoint/resume behavior.
The stats check guards against existing inputs, but is not an atomic lock:
do not run multiple launchers concurrently.
"""

import modal


app = modal.App("paul2026-swin-production-launcher")
FUNCTION_BY_SYSTEM = {
    "flat": "run_flat_production",
    "shared_hard": "run_shared_hard_production",
}


@app.local_entrypoint()
def main(seed: int, system: str = "flat"):
    if type(seed) is not int or seed not in (42, 123, 2026):
        raise ValueError("production seed must be one of: 42, 123, 2026")

    if type(system) is not str or system not in FUNCTION_BY_SYSTEM:
        raise ValueError("production system must be flat or shared_hard")
    function_name = FUNCTION_BY_SYSTEM[system]
    function = modal.Function.from_name("paul2026-swin", function_name)
    stats = function.get_current_stats()
    if stats.num_running_inputs > 0 or stats.backlog > 0:
        raise SystemExit(
            "production launch refused: "
            f"an existing {function_name} input is already running or queued"
        )

    call = function.spawn(seed)
    print("launch_status: submitted")
    print("app: paul2026-swin")
    print(f"system: {system}")
    print(f"function: {function_name}")
    print(f"seed: {seed}")
    print(f"function_call_id: {call.object_id}")
    print("local_process_required: false")
