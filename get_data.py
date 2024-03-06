"""Streamlit app. Data part"""
from typing import Union

import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

PROJECT_ID = "mortgage-model"
DATABASE_ID = "loans"
ORIGINATION_ID = "loan_origination"
PERFORMANCE_ID = "loan_performance"
PERFORMANCE_TRANSITION = "performance_transitions"
PERFORMANCE_AGGREGATION = "performance_aggregation"

missing_indicator = {
    "first_time_home_buyer_flag": "9",
    "mortgage_insurance_percent": 999,
    "number_of_units": 99,
    "occupancy_status": "9",
    "original_combined_ltv": 999,
    "original_dti": 999,
    "original_ltv": 999,
    "channel": "9",
    "property_type": "99",
    "zip_code": 0,
    "loan_purpose": "9",
    "number_of_borrowers": 99,
    "program_indicator": "9",
    "property_valuation_method": 9,
}


@st.cache_data(ttl=6000)
def run_query(query: str) -> pd.DataFrame:
    """run a query to pull data from bigquery and return a pandas data frame

    Args:
        query (str): query to be run

    Returns:
        pd.DataFrame: output data frame
    """
    query_job = client.query(query)
    rows_raw = query_job.result()
    return rows_raw.to_dataframe()


def format_name(name: Union[str, int]) -> str:
    """Formats a name, adding quotation marks if it's a string."""
    if isinstance(name, str):
        return f"'{name}'"
    return str(name)


def generate_group_list(var_list: list[str]) -> str:
    """convert a list of strings to a string used in select SQL"""
    group_list = []
    for var in var_list:
        if var in missing_indicator:
            group_list.append(
                f"nullif({var}, {format_name(missing_indicator[var])}) as {var}"
            )
        else:
            group_list.append(f"{var}")
    return ",\n".join(group_list)


def generate_grouping_list(var_list: list[str]) -> str:
    """convert a list of strings to a string used in 'grouping' SQL"""
    grouping_list = []
    for var in var_list:
        if var in missing_indicator:
            grouping_list.append(
                f"grouping(nullif({var}, {format_name(missing_indicator[var])})) as {var}_agg"
            )
        else:
            grouping_list.append(f"{var}_agg")
    return ",\n".join(grouping_list)


def previous_quarter(year_quarter: str) -> str:
    """find the previous quarter

    Args:
        year_quarter (str): string representation of this quarter "YYYYQX"

    Returns:
        str: string representation of previous quarter
    """
    year, quarter = year_quarter.split("Q")
    if quarter == "1":
        pre_year, pre_quarter = int(year) - 1, 4
    else:
        pre_year, pre_quarter = int(year), int(quarter) - 1
    return str(pre_year) + "Q" + str(pre_quarter)


@st.cache_data(ttl=600)
def get_base_summary(group_vars: list[str]) -> pd.DataFrame:
    """generate the base aggregation data.

    Args:
        group_vars (list[str]): the list of group variables for aggregation

    Returns:
        pd.DataFrame: the final data frame
    """
    query = f"""
    select
        format_date("%YQ%Q", origination_date) as quarter,
        {generate_group_list(group_vars[1:])},
        count(*) as loan_count,
        sum(original_upb) as upb_sum,
        avg(fico) as fico_avg,
        avg(nullif(original_ltv, {missing_indicator["original_ltv"]})) as ltv_avg,
        avg(nullif(original_dti, {missing_indicator["original_dti"]})) as dti_avg,
        grouping(format_date("%YQ%Q", origination_date)) as quarter_agg,
        {generate_grouping_list(group_vars[1:])}
    from `{PROJECT_ID}.{DATABASE_ID}.{ORIGINATION_ID}`
    group by grouping sets (1, (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), ()) 
    """
    return run_query(query)


# base_df = get_base_summary()
@st.cache_data
def generate_query_condition(vars_0: list[str], vars_1: list[str]) -> str:
    """generate query condition based on variables used in group statement 
    and variables not used in group statement

    Args:
        vars_0 (list[str]): list of variables used in the group statement in the query
        vars_1 (list[str]): list of variables not used in the group statement in the query

    Returns:
        str: a string of conditions used to filter the data from the base summary data frame
    """
    condition_list = []
    for var in vars_0:
        condition_list.append(f"{var}_agg==0")
    for var in vars_1:
        condition_list.append(f"{var}_agg==1")
    return " and ".join(condition_list)


@st.cache_data
def extract_quarter_metric(
    base_df: pd.DataFrame, metric: str, excluded_vars: list[str], quarter
) -> tuple[float, float]:
    """get the metric for this quarter and delta from last quarter

    Args:
        base_df (pd.DataFrame): base time series data frame
        metric (str): metric name / column name
        excluded_vars (list[str]): list of variables not used in the group statement
        quarter (_type_): quarter for the metric

    Returns:
        tuple[float, float]: metric for the quarter and delta between the last quarter
    """
    df = extract_timeseries_metric(base_df, metric, excluded_vars)
    count_this_quarter = df.loc[df["quarter"] == quarter, metric].iloc[0]
    count_pre_quarter = df.loc[df["quarter"] == previous_quarter(quarter), metric].iloc[
        0
    ]
    return count_this_quarter, count_this_quarter - count_pre_quarter


@st.cache_data
def extract_timeseries_metric(
    base_df: pd.DataFrame, metric: str, excluded_vars: list[str], segment_var: str = "None"
) -> pd.DataFrame:
    """extract a time series data with sepecific segment

    Args:
        base_df (pd.DataFrame): base summary data frame
        metric (str): metric name
        excluded_vars (list[str]): list of variables that are not used in the group statement
        segment_var (str, optional): the segment/group variable name. Defaults to "None".

    Returns:
        pd.DataFrame: the time series data frame
    """
    if segment_var == "None":
        query = generate_query_condition(["quarter"], excluded_vars)
        df = base_df.query(query)[["quarter", metric]]
    else:
        query = generate_query_condition(["quarter", segment_var], excluded_vars)
        df = base_df.query(query)[["quarter", metric, segment_var]]
    return df


@st.cache_data
def get_multi_dimension_summary(segment_vars: list[str], quarter: str) -> pd.DataFrame:
    """get multi-dimension summary for tree map chart

    Args:
        segment_vars (list[str]): the list of segment/group vars. It can include at most two vars.
        quarter (str): the quarter of interest

    Returns:
        pd.DataFrame 
    """
    if len(segment_vars) > 2:
        raise ValueError("You can only specify at most two segment variables")
    query = f"""
    select
        {generate_group_list(segment_vars)},
        count(*) as loan_count,
        sum(original_upb) as upb_sum,
        avg(fico) as fico_avg,
        avg(nullif(original_ltv, {missing_indicator["original_ltv"]})) as ltv_avg,
        avg(nullif(original_dti, {missing_indicator["original_dti"]})) as dti_avg
    from `{PROJECT_ID}.{DATABASE_ID}.{ORIGINATION_ID}`
    where format_date("%YQ%Q", origination_date) = "{quarter}"
    group by {", ".join([str(i) for i in range(1, len(segment_vars) + 1)])}
    """
    return run_query(query)


@st.cache_data
def pull_data_from_bq(table_id: str) -> pd.DataFrame:
    """pull data from bigquery as is and return pandas dataframe

    Args:
        table_id (str): the tabel id in bigquery

    Returns:
        pd.DataFrame: 
    """
    query = f"""
    select
    *
    FROM
    `{PROJECT_ID}.{DATABASE_ID}.{table_id}`
    order by performance_month
    """
    return run_query(query)

@st.cache_data
def get_delinquency_rate(
    df_aggregation: pd.DataFrame, group_vars: list[str]
) -> pd.DataFrame:
    """get delinqeucy rate for the specific group variables

    Args:
        df_aggregation (pd.DataFrame): base performance summary data
        group_vars (list[str]): list variables used in group by statement

    Returns:
        pd.DataFrame: final data frame
    """
    all_group_vars = [
        "performance_month",
        "number_of_units",
        "first_time_home_buyer_flag",
        "occupancy_status",
        "channel",
        "property_type",
        "loan_purpose",
        "number_of_borrowers",
        "property_state",
        "vintage",
    ]
    non_group_vars = [var for var in all_group_vars if var not in group_vars]
    df_filtered = df_aggregation.query(
        generate_query_condition(group_vars, non_group_vars)
    )
    return df_filtered
