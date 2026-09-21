"""Submit a locked evaluation phase to the deployed app and exit immediately.

The busy check is advisory, not an atomic distributed mutex. Submit only one
launcher at a time. This launcher never submits any training function.
"""

import modal


app = modal.App("paul2026-swin-evaluation-launcher")
FUNCTION_BY_MODE = {
    "preflight": "run_isic_evaluation_preflight",
    "isic": "run_isic_evaluation_production",
    "hiba-preflight": "run_hiba_evaluation_preflight",
    "hiba": "run_hiba_evaluation_production",
}
BUSY_FUNCTIONS = (*FUNCTION_BY_MODE.values(), "run_flat_production", "run_shared_hard_production", "run_flat_smoke")


@app.local_entrypoint()
def main(mode: str):
    if type(mode) is not str or mode not in FUNCTION_BY_MODE:
        raise ValueError("evaluation mode must be preflight, isic, hiba-preflight, or hiba")
    functions = {}
    for name in BUSY_FUNCTIONS:
        function = modal.Function.from_name("paul2026-swin", name)
        stats = function.get_current_stats()
        if stats.num_running_inputs > 0 or stats.backlog > 0:
            raise SystemExit(f"evaluation launch refused: {name} is running or queued")
        functions[name] = function
    function = functions[FUNCTION_BY_MODE[mode]]
    call = function.spawn()
    print("launch_status: submitted")
    print("app: paul2026-swin")
    print(f"mode: {mode}")
    print(f"function: {FUNCTION_BY_MODE[mode]}")
    print(f"function_call_id: {call.object_id}")
    print("local_process_required: false")
