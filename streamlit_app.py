"""Streamlit app. Layout part"""

from dataclasses import dataclass
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from get_data import (
    extract_quarter_metric,
    extract_timeseries_metric,
    get_base_summary,
    get_multi_dimension_summary,
    get_delinquency_rate,
    pull_data_from_bq,
)

st.set_page_config(page_title="Dashboard Demo", layout="wide")

# list of all possible groupby vars
group_vars = [
    "quarter",
    "number_of_units",
    "first_time_home_buyer_flag",
    "occupancy_status",
    "channel",
    "property_type",
    "loan_purpose",
    "number_of_borrowers",
]

PERFORMANCE_TRANSITION = "performance_transitions"
PERFORMANCE_AGGREGATION = "performance_aggregation"


@dataclass
class Metric:
    """data class of metrics"""

    name: str
    title: str


def create_line_fig(
    df: pd.DataFrame, x: str, y: str, seg_var: str = None, **kwargs
) -> go.Figure:
    """create a line figure

    Args:
        df (pd.DataFrame): source data frame
        x (str): column name of x-axis
        y (str): column name of y-axis
        seg_var (str, optional): groupby variable name. Defaults to None.

    Returns:
        go.Figure: Plotly figure
    """
    if seg_var == "None":
        df = df.sort_values(x)
        fig = go.Figure(
            data=go.Scatter(x=df[x], y=df[y]),
            layout=kwargs,
        )
    else:
        fig = go.Figure()
        for seg in df[seg_var].drop_duplicates():
            temp_df = df.loc[df[seg_var] == seg].sort_values(x)
            fig.add_trace(go.Scatter(x=temp_df[x], y=temp_df[y], name=str(seg)))
        fig.update_layout(kwargs)
    return fig


def create_bar_fig(
    df: pd.DataFrame, x: str, y: str, seg_var: str = None, **kwargs
) -> go.Figure:
    """create a bar fig

    Args:
        df (pd.DataFrame): source data frame
        x (str): column name of x-axis
        y (str): column name of y-axis
        seg_var (str, optional): column name of groupby variable. Defaults to None.

    Returns:
        go.Figure: Plotly figure
    """
    if seg_var == "None":
        df = df.sort_values(x)
        fig = go.Figure(
            data=go.Bar(x=df[x], y=df[y]),
            layout=kwargs,
        )
    else:
        fig = go.Figure()
        for seg in df[seg_var].drop_duplicates():
            temp_df = df.loc[df[seg_var] == seg].sort_values(x)
            fig.add_trace(go.Bar(x=temp_df[x], y=temp_df[y], name=str(seg)))
        fig.update_layout(kwargs)
    return fig


@st.cache_data
def generate_source_df(df: pd.DataFrame, segment: str, x: str, y: str) -> pd.DataFrame:
    """generate filted data frame out of original data frame

    Args:
        df (pd.DataFrame): original data frame
        segment (str): column name of groupby variable
        x (str): column name of x-axis
        y (str): column name of y-axis

    Returns:
        pd.DataFrame: filted data frame
    """
    if segment == "None" and x == "quarter":
        return extract_timeseries_metric(df, y, group_vars[1:])
    if x == "quarter":
        group_vars_excluded = [var for var in group_vars[1:] if var != segment]
        return extract_timeseries_metric(df, y, group_vars_excluded, segment)
    if x != "quarter":
        group_vars_excluded = [var for var in group_vars if var != x]
        return extract_timeseries_metric(df, y, group_vars_excluded, segment)
    return None


@dataclass
class Chart:
    """data class of a chart"""

    source_df: pd.DataFrame
    x: str
    y: str
    chart_type: str
    group: str
    layout: dict

    @property
    def fig(self) -> go.Figure:
        """fig property of Chart class.
        It creates a plotly figure based on data frame, chart type and layout"""
        if self.chart_type == "Line":
            fig = create_line_fig(
                self.source_df, self.x, self.y, seg_var=self.group, **self.layout
            )
        elif self.chart_type == "Bar":
            fig = create_bar_fig(
                self.source_df, self.x, self.y, seg_var=self.group, **self.layout
            )
        elif self.chart_type[:3] == "Bar":
            barmode = self.chart_type.split("-")[1].lower()
            self.layout.update({"barmode": barmode})
            fig = create_bar_fig(
                self.source_df, self.x, self.y, seg_var=self.group, **self.layout
            )
        if self.group != "None":
            fig.update_layout({"legend_title_text": self.group})
        return fig


st.title("Freddie Mac Loans Origination and Performance")
md = """
- This dashboard uses part of Freddie Mac public [loan level data sets](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset). 
- This dashboard mainly uses [Streamlit](https://streamlit.io/) and Google Cloud Bigquery.
- The dashboard does not reflect actual Freddie Mac's business since the sample used does not reflect the whole portfolio.
- The dashboard cannot be used for any risk management purposes.
- Created by [The Model Advantage](https://www.themodeladvantage.com/)
"""
with st.expander("**Note and Disclaimer**"):
    st.markdown(md)

tab1, tab2 = st.tabs(["Loan Origination", "Loan Performance"])

base_df = get_base_summary(group_vars)

with tab1:
    metric_container = st.container()  # metrics
    overall_container = st.container()  # metrics over time
    segment_container = st.container()  # metrics decomposition

with metric_container:
    col = st.columns(3)
    with col[0]:
        st.write("# Snapshot for Quarter")
        quarter = st.selectbox(
            "Select Quarter",
            base_df.sort_values("quarter", ascending=False)[
                "quarter"
            ].drop_duplicates(),
        )
    col1, col2, col3 = st.columns(3)

# Metrics for the quarter snapshot
# create three metrics
loan_count = Metric(name="loan_count", title="Total Loan Count")
upb = Metric(name="upb_sum", title="Total UPB")
fico = Metric(name="fico_avg", title="Average FICO")
for col, metric in zip([col1, col2, col3], [loan_count, upb, fico]):
    with col:
        metric_value, delta = extract_quarter_metric(
            base_df, metric.name, group_vars[1:], quarter
        )
        st.metric(metric.title, int(metric_value), int(delta))

with overall_container:
    st.write("# Overall Trend")
    # chart type and segmentation variable selectors
    col_segmentation = st.columns(3)
    # three trend charts
    col_count, col_upb, col_fico = st.columns(3)


# segmentation variable and chart type selectors
def build_selection_layer(key: str) -> tuple[str, str]:
    """create a selection layer including a select box and radio selector

    Args:
        key (str): key to differentiate componentors

    Returns:
        tuple[str, str]: two strings holding values of select box and radio selector
    """
    groupby_var = st.selectbox(
        "Select Group Variable",
        ["None"] + group_vars[1:],
        key=key + "_selectbox",
    )
    if groupby_var == "None":  # no segmentation
        chart_options = ("Line", "Bar")
    else:
        chart_options = ("Line", "Bar-Group", "Bar-Stack")
    chart_type = st.radio(
        "Select Chart Type 👇", chart_options, horizontal=True, key=key + "_radio"
    )
    return groupby_var, chart_type


with col_segmentation[0]:
    selected_groupby, selected_chart_type = build_selection_layer(key="origination")

# create count, upb, and fico chart over time
for col, metric, title in zip(
    [col_count, col_upb, col_fico],
    ["loan_count", "upb_sum", "fico_avg"],
    ["Total Loan Count", "Total Original UPB", "Average FICO"],
):
    source_df = generate_source_df(base_df, selected_groupby, "quarter", metric)
    chart = Chart(
        source_df=source_df,
        x="quarter",
        y=metric,
        chart_type=selected_chart_type,
        group=selected_groupby,
        layout=dict(
            title={"text": title, "font_size": 20},
            # width=1200,
            xaxis_title={"text": "Quarter"},
        ),
    )
    with col:
        st.plotly_chart(chart.fig, use_container_width=True)

# segment container
# snapshot at quarter by segment variable

with segment_container:
    st.write("# Snapshot by Segment")
    quarter1 = st.selectbox(
        "Select Quarte:",
        base_df.sort_values("quarter", ascending=False)["quarter"].drop_duplicates(),
    )
    metric = st.radio("Count or UPB", ("loan_count", "upb_sum"))
    seg_vars = st.multiselect(
        "Select **Two** Group Variables:",
        group_vars,
        default=["property_type", "channel"],
    )
    if len(seg_vars) < 2:
        st.stop()
    elif len(seg_vars) > 2:
        st.error("You can only choose TWO group variables!")
        st.stop()
    df_seg_var1 = get_multi_dimension_summary([seg_vars[0]], quarter1)
    df_seg_var2 = get_multi_dimension_summary([seg_vars[1]], quarter1)
    df_seg_vars = get_multi_dimension_summary(seg_vars, quarter1)
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = px.bar(
            df_seg_var1,
            y=seg_vars[0],
            x=metric,
            orientation="h",
            color="fico_avg",
            hover_data=["loan_count"],
            color_continuous_scale="RdBu",
            color_continuous_midpoint=np.average(
                df_seg_var1["fico_avg"], weights=df_seg_var1[metric]
            ),
        )
        fig1.update_layout(
            {"title": {"text": f"{metric.upper()} by {seg_vars[0]}", "font_size": 20}}
        )
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.treemap(
            df_seg_vars,
            path=[px.Constant("All"), *seg_vars],
            values=metric,
            color="fico_avg",
            hover_data=["loan_count"],
            color_continuous_scale="RdBu",
            color_continuous_midpoint=np.average(
                df_seg_vars["fico_avg"], weights=df_seg_vars[metric]
            ),
        )
        fig2.update_layout(margin={"t": 50, "l": 25, "r": 25, "b": 25})
        fig2.update_layout(
            {
                "title": {
                    "text": f"{metric.upper()} by {seg_vars[0]} and {seg_vars[1]}",
                    "font_size": 20,
                }
            }
        )
        st.plotly_chart(fig2)
    with col3:
        fig3 = px.bar(
            df_seg_var2,
            y=seg_vars[1],
            x=metric,
            orientation="h",
            color="fico_avg",
            hover_data=["loan_count"],
            color_continuous_scale="RdBu",
            color_continuous_midpoint=np.average(
                df_seg_var2["fico_avg"], weights=df_seg_var2[metric]
            ),
        )
        fig3.update_layout(
            {"title": {"text": f"{metric.upper()} by {seg_vars[1]}", "font_size": 20}}
        )
        st.plotly_chart(fig3)


# performance tab

transitions = pull_data_from_bq(PERFORMANCE_TRANSITION)

with tab2:
    st.write("# Deep Delinquency Status Transitions")
    col1, col2, col3 = st.columns([1, 2, 2])
    transition_col1, transition_col2, transition_col3 = st.columns([1, 3, 1])
    st.write("# Performance Over Time")
    selection_layer = st.columns([1, 4])
    trend_col1, trend_col2, trend_col3 = st.columns(3)
    st.write("# Performance Over State")
    geographic_selection_layer = st.columns([10, 1, 10])
    geographic_layer = st.columns([10, 1, 10])


@st.cache_data
def prepare_data_for_sankey(
    df: pd.DataFrame, begin_month: date, month_counts: int
) -> pd.DataFrame:
    """filter and prepare data frame for Sankey table. Remove delinquency status
    0, 1, 2, 3 since they have too many volumes that makes the transitions between
    deep delinquency status unnoticable.

    Args:
        df (pd.DataFrame): original data frame
        begin_month (date): the beginning month of the transition
        month_counts (int): number of months to track

    Returns:
        pd.DataFrame: the filtered data frame and ready for Sankey
    """
    df_filtered = df.loc[
        (df["performance_month"] >= begin_month)
        & (df["performance_month"] <= begin_month + relativedelta(months=month_counts))
    ]
    # remove dlq status up to D90, MC and New Originations
    df_filtered = df_filtered.loc[
        ~(df_filtered["dlq"].isin(["0", "1", "2", "3", "New Origination", "MC"]))
    ]
    df_filtered["dlq_next"] = np.where(
        df_filtered["dlq_next"].isin(["0", "1", "2", "3"]),
        "Recover",
        df_filtered["dlq_next"],
    )
    return df_filtered


def plot_sankey(df: pd.DataFrame) -> go.Figure:
    """generate a sankey chart from the data frame

    Args:
        df (pd.DataFrame): the data frame used for the chart

    Returns:
        go.Figure: plotly Figure
    """
    # get the list of months covered in the chart
    months = sorted(df["performance_month"].drop_duplicates().to_list())
    tempdf = df.copy()
    n_month = len(months)
    # the ending month in the sankey diagram
    last_month = months[n_month - 1] + relativedelta(months=1)
    # node list and node mapping. The nodes for this month include nodes in the starting stauts (column 'dlq') of this month
    # and nodes in the ending status of the previous month (column 'dlq_next')
    # The nodes for the last month is the nodes in the ending status of the previous month.
    node_labels = []
    node_mappings = []
    for idx, month in enumerate(months):
        if idx == 0:
            node_labels.append(
                df.loc[df["performance_month"] == month, "dlq"]
                .drop_duplicates()
                .sort_values()
                .to_list()
            )
        else:
            previous_month = months[idx - 1]
            temp_node = (
                df.loc[df["performance_month"] == previous_month, "dlq_next"]
                .drop_duplicates()
                .to_list()
            )
            temp_node = sorted(
                list(
                    set(
                        temp_node
                        + df.loc[df["performance_month"] == month, "dlq"]
                        .drop_duplicates()
                        .to_list()
                    )
                )
            )
            node_labels.append(temp_node)
    node_labels.append(
        df.loc[df["performance_month"] == months[n_month - 1], "dlq_next"]
        .drop_duplicates()
        .sort_values()
        .to_list()
    )
    # creating node: index mapping
    for idx, node_label in enumerate(node_labels):
        node_mappings.append(
            {
                node: i + sum(list(map(len, node_labels[:idx])))
                for i, node in enumerate(node_label)
            }
        )
    # create source and target columns used in Sankey diagram
    # source and target should be the positions of nodes
    for idx, month in enumerate(months):
        tempdf.loc[tempdf["performance_month"] == month, "source"] = tempdf.loc[
            tempdf["performance_month"] == month, "dlq"
        ].map(node_mappings[idx])
        tempdf.loc[tempdf["performance_month"] == month, "target"] = tempdf.loc[
            tempdf["performance_month"] == month, "dlq_next"
        ].map(node_mappings[idx + 1])
    # create labels, and x, y positions
    # label is the list of all node_labels

    label = []
    x = []
    y = []
    for idx, node_label in enumerate(node_labels):
        label += node_label
        x += [0 + 1 / n_month * idx] * len(node_label)
        if idx == 0:
            y += [i / len(node_label) for i in range((len(node_label)))]
        else:
            y += [1 - i / len(node_label) for i in range((len(node_label)))]
    # ready to plot the chart
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                # domain=dict(x=(0,1), y=(0,1)),
                node=dict(
                    pad=15,
                    thickness=15,
                    line=dict(color="black", width=0.8),
                    label=label,
                    x=x,
                    y=y,
                ),
                link=dict(
                    source=tempdf["source"].to_list(),
                    target=tempdf["target"].to_list(),
                    value=tempdf["count"].to_list(),
                ),
            )
        ],
        layout={
            "title": {
                "text": f"Deep Delinquency Status Transition between {months[0].strftime('%Y-%m')} and {last_month.strftime('%Y-%m')}"
            },
            "title_font_size": 20,
            "margin": {"r": 3},
        },
    )
    for idx, month in enumerate(months):
        fig.add_annotation(
            x=1 / n_month * idx, y=1, text=month.strftime("%Y-%m"), showarrow=False
        )
    fig.add_annotation(x=1, y=1, text=last_month.strftime("%Y-%m"), showarrow=False)
    fig.update_annotations(font=dict(family="Courier New", size=20, color="black"))
    return fig


month_options = (
    transitions.query("performance_month>='2020-01-01'")["performance_month"]
    .drop_duplicates()
    .to_list()
)
with transition_col1:
    starting_month = st.selectbox(
        "Select starting month",
        month_options,
        format_func=lambda x: date.strftime(x, format="%Y-%m"),
    )
    number_month = st.radio(
        "Select Number of Months", (2, 3, 4, 5), index=0, horizontal=True
    )

transitions_filtered = prepare_data_for_sankey(
    transitions, starting_month, number_month - 1
)
with transition_col2:
    st.plotly_chart(plot_sankey(transitions_filtered), use_container_width=True)

# D120/D150/D180+/REOFA over time
# Metrics over time by buckets (line, bar)
# REOFA/Deep delinquency rate across state change by time (map)
# Snapshot, scatter plots
with selection_layer[0]:
    segment_var_performance, chart_type_performance = build_selection_layer(
        key="performance"
    )

performance_aggregation_base = pull_data_from_bq(PERFORMANCE_AGGREGATION)
if segment_var_performance == "None":
    dlq_over_time = get_delinquency_rate(
        performance_aggregation_base, ["performance_month"]
    )
else:
    dlq_over_time = get_delinquency_rate(
        performance_aggregation_base, ["performance_month", segment_var_performance]
    )

# create performance charts over time, by group var
for col, metric in zip(
    [trend_col1, trend_col2, trend_col3], ["d120", "d180_puls", "reofa"]
):
    with col:
        chart = Chart(
            source_df=dlq_over_time,
            x="performance_month",
            y=metric,
            chart_type=chart_type_performance,
            group=segment_var_performance,
            layout=dict(
                title={"text": f"{metric.upper()} Rate over Time", "font_size": 20},
                # width=1200,
                xaxis_title={"text": "Month"},
            ),
        )
        st.plotly_chart(chart.fig)

geographic_data = get_delinquency_rate(
    performance_aggregation_base, ["performance_month", "property_state"]
)
# Load the GeoJSON file for US states
GEOJSON = "https://raw.githubusercontent.com/python-visualization/folium/master/tests/us-states.json"

with geographic_selection_layer[0]:
    selected_month = st.select_slider(
        "Select Performance Month",
        options=month_options,
        format_func=lambda x: date.strftime(x, format="%Y-%m"),
    )
    selected_metric = st.radio(
        "Select Metric",
        options=["d120", "d150", "d180_puls", "reofa"],
        format_func=str.upper,
        horizontal=True,
        index=0,
    )
    # st.write("Start time:", selected_month)

with geographic_layer[0]:
    geo_df = geographic_data[geographic_data["performance_month"] == selected_month]
    # st.write(geo_df)
    # Create Choropleth map
    map_fig = go.Figure(
        go.Choropleth(
            geojson=GEOJSON,
            locations=geo_df["property_state"],
            z=geo_df[selected_metric],
            locationmode="USA-states",
            colorscale="Peach",
            colorbar_title=selected_metric.upper(),
        ),
        layout={"height": 600},
    )

    # Update layout
    map_fig.update_layout(
        title=f"{selected_metric.upper()} per State",
        title_font_size=20,
        geo_scope="usa",
    )
    st.plotly_chart(map_fig, use_container_width=True)

with geographic_layer[2]:
    top_10 = (
        geo_df[["property_state", selected_metric]]
        .sort_values(selected_metric, ascending=False)
        .head(10)
    )
    bar_fig = go.Figure(
        data=go.Bar(
            x=top_10[selected_metric],
            y=top_10["property_state"],
            orientation="h",
        ),
        layout={
            "height": 600,
            "title": {
                "text": f"Top 10 States with The Largest {selected_metric.upper()} Rate",
                "font_size": 20,
            },
            "yaxis": {"autorange": "reversed"},
        },
    )
    st.plotly_chart(bar_fig, use_container_width=True)
