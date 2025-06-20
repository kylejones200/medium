
import pandas as pd
import numpy as np

np.random.seed(42)
months = pd.date_range("2022-01-01", periods=12, freq="M")
regions = ["North", "South", "West"]
data = {
    "Month": np.tile(months, len(regions)),
    "Region": np.repeat(regions, len(months)),
    "Sales": np.random.randint(90, 150, size=len(months) * len(regions))
}
df = pd.DataFrame(data)


# Time Series Plot with Matplotlib and Seaborn

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Month", y="Sales", hue="Region", marker="o")
plt.title("Monthly Sales by Region", fontsize=14, weight="bold")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("sales_seaborn.png", dpi=300)
plt.show()


# Interactive Plot with Plotly

import plotly.express as px

fig = px.line(df, x="Month", y="Sales", color="Region",
              markers=True, title="Monthly Sales by Region")
fig.update_layout(title_font=dict(size=18), template="simple_white")
fig.show()

# Interactive Plot with Altair
import altair as alt

alt_df = df[df["Month"].dt.month == 6]  # Filter to show June sales
bar = alt.Chart(alt_df).mark_bar().encode(
    x=alt.X("Region:N", title="Region"),
    y=alt.Y("Sales:Q", title="Sales"),
    color="Region:N"
).properties(title="June Sales by Region")
text = bar.mark_text(
    align="center", baseline="bottom", dy=-2
).encode(text="Sales:Q")
(bar + text).configure_view(stroke=None).show()



#### Streamlit Dashboard

# Save the following as app.py and run with streamlit run app.py:

import streamlit as st

st.title("Regional Sales Dashboard")
selected_region = st.selectbox("Select Region", df["Region"].unique())
filtered = df[df["Region"] == selected_region]
st.line_chart(filtered.set_index("Month")["Sales"])
st.metric("Average Sales", f"{filtered['Sales'].mean():.2f}")
st.metric("Max Sales", f"{filtered['Sales'].max()}")
if st.checkbox("Show raw data"):
    st.write(filtered)
