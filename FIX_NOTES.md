# Income Level Prediction Fix Notes

## Fixed

- Added Python 3.12 deployment pinning and compatible binary-wheel dependencies.
- Made external artifact paths relative to `app.py`.
- Added clear download failure handling for Google Drive model and scaler files.
- Added artifact type/schema validation before the app starts.
- Added a check that the scaler contains the four expected numerical columns.
- Added a check that the model feature count matches the app's explicit training schema.

## Required external artifacts

The repository downloads these files from Google Drive:

```text
income_prediction_rf_model.joblib
income_prediction_scaler.joblib
```

Both Drive files must be shared as “Anyone with the link” and must remain compatible with the schema in `app.py`.

## Remaining model-quality limitation

The app still duplicates the notebook's one-hot preprocessing in `TRAINING_COLUMNS`. The production-grade follow-up is to export one complete sklearn preprocessing/model pipeline and remove the duplicated manual schema.
