import pandas as pd
import plotly.express as px

def plotly_pie_3d_style(
    counts,
    labels,
    title,
    explode_label=None,
    save_path=None,
):
    df = pd.DataFrame({"label": labels, "count": counts})
    df["pull"] = 0.0

    if explode_label is not None:
        df.loc[df["label"] == explode_label, "pull"] = 0.12

    fig = px.pie(
        df,
        values="count",
        names="label",
        hole=0.4,  # donut = pseudo 3D
        color="label",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=title,
    )

    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        pull=df["pull"],
        marker=dict(line=dict(color="black", width=1)),
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    if save_path:
        fig.write_image(save_path, width=800, height=600)
    else:
        fig.show()



counts = [10, 150, 20, 5]   # example
labels = ["0", "1", "2", "Other"]
plotly_pie_3d_style(
    counts,
    labels,
    title="Pie Chart for Single-Channel Image",
    explode_label="2",              # detach slice for class "2"
    save_path="chart.png"           # optional
)
