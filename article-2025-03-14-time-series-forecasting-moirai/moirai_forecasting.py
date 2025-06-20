import pandas as pd
import matplotlib.pyplot as plt
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

SIZE = "small"  # Model size
PDT = 64        # Prediction length
CTX = 200       # Context length
PSZ = "auto"    # Patch size
BSZ = 32        # Batch size
TEST = 64       # Test set length

# Load data
df = pd.read_csv("Ercot_Native_Load_2025 (1).csv")
df["timestamp"] = pd.to_datetime(df["Date"])
df.set_index("timestamp", inplace=True)
df["ERCOT"] = pd.to_numeric(df["ERCOT"], errors="coerce")
df = df.asfreq("h")  # Hourly frequency

# Convert into GluonTS dataset
ds = PandasDataset(dict(df[["ERCOT"]]))

# Train-test split
train, test_template = split(ds, offset=-TEST)
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# Model setup
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

# Create predictor and forecast
predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

# Visualization
input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)

plot_single(
    inp,
    label,
    forecast,
    context_length=CTX,
    name="moirai_forecast",
    show_label=True,
)
plt.show()
