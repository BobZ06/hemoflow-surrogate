# Serving API and interactive demo verification

**Primary owner:** Letian

This verification covers `src/hemoflow/serving/api.py`,
`src/hemoflow/serving/demo.py`, and
`src/hemoflow/serving/static/demo.html`.

## Why the prediction API and demo are separate

`api.py` provides the general prediction service. Its `/predict` endpoint accepts
a vessel description and returns a compact prediction summary. This interface
can be used by programs without depending on the interactive web page.

`demo.py` provides behavior needed specifically by the interactive demo. It
returns the predicted field, reference field, baseline field, vessel geometry,
summary statistics, and timing information in one request.

`static/demo.html` provides the user interface. It reads the sliders, sends
requests to `/demo/predict`, and draws the vessel and wall-shear-stress fields.

Keeping these parts separate prevents display-specific requirements from
changing the general `/predict` interface. The demo page can be changed or
removed without changing the main prediction endpoint.

## Service startup

The service was started locally with:

`.\.venv\Scripts\python.exe -m uvicorn hemoflow.serving.api:app --host 127.0.0.1 --port 1000`

Port `1000` was used because port `8000` was already occupied.

The status endpoint returned HTTP `200` with:

- Ready: `true`
- Model: `mlp`
- Run ID: `mlp-53e1ed5fd3`
- Parameters: `35,585`
- Grid: `128 × 64`

A valid request to `/predict` also returned HTTP `200` and produced a complete
WSS prediction.

## Physics-prior toggle

The web page sends the value of `physics_prior` to `/demo/predict`.

When `physics_prior` is `true`, `demo.py` selects the checkpoint trained with
`cfg.model.physics_residual = true`. The network predicts a dimensionless
correction ratio, and the Poiseuille field supplies the absolute WSS scale.

When `physics_prior` is `false`, `demo.py` selects the ablation checkpoint
trained with `cfg.model.physics_residual = false`.

Both checkpoints use the MLP architecture, so both responses report
`model_name = "mlp"`. The `physics_prior` value and prediction behavior identify
the active checkpoint mode.

The same vessel request produced:

- Physics prior on:
  - `physics_prior = true`
  - Model: `mlp`
  - Peak WSS: `6.0545 Pa`
  - Mean WSS: `1.8494 Pa`

- Physics prior off:
  - `physics_prior = false`
  - Model: `mlp`
  - Peak WSS: `4.8887 Pa`
  - Mean WSS: `1.5205 Pa`

Both requests returned HTTP `200`. The changed prediction values confirm that
the toggle changes the model behavior rather than only changing a page label.

## Invalid input handling

Three invalid requests were sent to `/predict`.

### Too few radius samples

Input: `radii_mm = [2.0, 2.0, 2.0]`

Result: HTTP `422`

Message: `List should have at least 4 items after validation, not 3`

### Negative radius

Input: `radii_mm = [2.0, -1.0, 2.0, 2.0]`

Result: HTTP `422`

Message: `Value error, every radius must be positive`

### Implausible radius range

Input: `radii_mm = [0.001, 10.0, 0.001, 10.0]`

Result: HTTP `422`

Message: `Value error, radius profile spans an implausible range; check units (mm expected)`

All three invalid requests were rejected with clear validation messages. The
service did not return a prediction for invalid input.

## Offline cold-start verification

The running service was stopped, Wi-Fi was disabled, and the service was
started again from a new process.

The page at `http://127.0.0.1:1000/` loaded successfully without internet
access. The controls, prediction requests, field displays, and physics-prior
toggle continued to work.

This confirms that the demo uses local static files, local Python code, and local
checkpoints. It does not require an external network connection during use.

## Verification status

- [x] Explained why `/predict` and the interactive demo are separate.
- [x] Verified a valid `/predict` request returns HTTP `200`.
- [x] Verified the physics-prior toggle in both states.
- [x] Confirmed that the toggle changes prediction values.
- [x] Tested three invalid inputs.
- [x] Confirmed all invalid inputs returned clear HTTP `422` responses.
- [x] Cold-started the service with Wi-Fi disabled.
- [x] Confirmed the full demo works without internet access.
