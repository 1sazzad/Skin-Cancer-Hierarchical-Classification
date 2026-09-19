"""Submit production to an already deployed app and exit without waiting.

Deploy code (does not start training):
    modal deploy scripts/modal_paul2026_swin.py
Submit once, without --detach:
    modal run scripts/launch_paul2026_swin_production.py
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


@app.local_entrypoint()
def main():
    function = modal.Function.from_name("paul2026-swin", "run_flat_production")
    stats = function.get_current_stats()
    if stats.num_running_inputs > 0 or stats.backlog > 0:
        raise SystemExit(
            "production launch refused: "
            "an existing run_flat_production input is already running or queued"
        )

    call = function.spawn()
    print("launch_status: submitted")
    print("app: paul2026-swin")
    print("function: run_flat_production")
    print(f"function_call_id: {call.object_id}")
    print("local_process_required: false")
