"""Non-blocking COM-04 launcher; submit only one launcher at a time.

Busy checks across both phases are advisory, not a distributed mutex.
Wait for successful CPU preflight completion before submitting evaluate.
"""
import modal

APP_NAME = "comesyso2026-probability-fusion"
app = modal.App("comesyso2026-internal-isic-launcher")
FUNCTION_BY_MODE = {
    "preflight": "run_internal_isic_preflight",
    "evaluate": "run_internal_isic_production",
}
BUSY_FUNCTIONS = (*FUNCTION_BY_MODE.values(), "run_validation_fit_preflight", "run_validation_fit_production")


@app.local_entrypoint()
def main(mode: str):
    if type(mode) is not str or mode not in FUNCTION_BY_MODE:
        raise ValueError("COM-04 mode must be preflight or evaluate")
    functions = {}
    for name in BUSY_FUNCTIONS:
        function = modal.Function.from_name(APP_NAME, name)
        stats = function.get_current_stats()
        if stats.num_running_inputs > 0 or stats.backlog > 0:
            raise SystemExit(f"launch refused: {name} is running or queued")
        functions[name] = function
    call = functions[FUNCTION_BY_MODE[mode]].spawn()
    print("launch_status: submitted")
    print(f"app: {APP_NAME}")
    print(f"mode: {mode}")
    print(f"function: {FUNCTION_BY_MODE[mode]}")
    print(f"function_call_id: {call.object_id}")
    print("local_process_required: false")
