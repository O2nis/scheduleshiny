# app.py — Shiny (Python) rewrite of the Streamlit scheduler20.py
# Notes:
# - Keeps all core features: file/template workflow, create-table workflow, discipline config,
#   man-hours/cost aggregations, heatmaps, stacked histogram, donut, cost by period/task,
#   cumulative cash flow, and invoice schedule export.
# - UI/UX differs where Streamlit had widgets without direct Shiny equivalents (e.g., data_editor, color picker).
#   Here we implement dynamic numeric inputs for utilization editing and a hex color text field for discipline colors.
#
# How to run:
#   pip install shiny matplotlib seaborn pandas numpy python-dateutil
#   shiny run --reload app.py

from __future__ import annotations

import io
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple

import matplotlib as mpl
mpl.use("Agg")  # ensure headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from shiny import App, reactive, render, ui

# ------------------------------
# Utility functions (ported)
# ------------------------------

def create_template_csv() -> str:
    template_data = {
        "Service": [f"Service {i}" for i in range(1, 6)],
        "Role": [
            "Electrical Engineer",
            "Civil Engineer",
            "Mechanical Engineer",
            "HSE",
            "Project Manager",
        ],
        "Hourly Cost": [50, 60, 70, 80, 90],
    }
    for i in range(1, 37):
        template_data[str(i)] = [round(random.uniform(0, 0.9), 1) for _ in range(5)]
    df = pd.DataFrame(template_data)
    return df.to_csv(index=False)


def get_period_start_end_dates(period_number: int, project_start: datetime, time_period: str):
    if time_period == "Months":
        start_date = project_start + relativedelta(months=period_number - 1)
        end_date = start_date + relativedelta(months=1) - timedelta(days=1)
    else:
        start_date = project_start + timedelta(weeks=period_number - 1)
        end_date = start_date + timedelta(days=6)
    return start_date, end_date


def map_discipline(role: str, discipline_keywords: Dict[str, Dict[str, List[str]]]) -> str:
    role_lower = str(role).lower()
    for discipline, data in discipline_keywords.items():
        for keyword in data.get("keywords", []):
            if keyword.lower() in role_lower:
                return discipline
    return "Other"


def get_earliest_period(data: pd.DataFrame, index_col: str, period_columns: List[str]) -> Dict[str, int]:
    earliest_periods: Dict[str, int] = {}
    for idx in data[index_col].unique():
        idx_data = data[data[index_col] == idx]
        for col in period_columns:
            period_sum = idx_data[col].sum()
            if pd.notna(period_sum) and period_sum > 0:
                earliest_periods[idx] = int(col)
                break
        else:
            earliest_periods[idx] = 10**9
    return earliest_periods


# ------------------------------
# Default disciplines
# ------------------------------
DEFAULT_DISCIPLINES = {
    "Civil": {"keywords": ["Civil", "GIS", "Geotechnical"], "color": "#1f77b4"},
    "Electrical": {"keywords": ["Electrical", "PV"], "color": "#2ca02c"},
    "HSE": {"keywords": ["HSE", "Safety"], "color": "#d62728"},
    "Instrument": {"keywords": ["Instrument", "SCADA", "I&C", "Automation"], "color": "#9467bd"},
    "Management": {"keywords": ["Project Manager", "DCC", "Project Engineer", "Manager"], "color": "#ff7f0e"},
    "Mechanical": {"keywords": ["Mechanical"], "color": "#e377c2"},
    "Site Supervision": {"keywords": ["Supervisor", "Commissioning"], "color": "#7f7f7f"},
    "Other": {"keywords": [], "color": "#bcbd22"},
}

# ------------------------------
# Shiny UI
# ------------------------------

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Data Input Method"),
        ui.input_radio_buttons(
            "input_method", "Choose Input Method", ["Create Table", "Upload File"], selected="Create Table"
        ),
        ui.input_text("currency", "Enter Currency Symbol", value="$"),
        ui.panel_conditional(
            "input.input_method === 'Upload File'",
            ui.input_file("uploaded", "Upload Schedule CSV or Excel", multiple=False, accept=[".csv", ".xlsx"]),
            ui.input_action_button("download_template", "Download CSV Template"),
            ui.help_text("Click the button to download a CSV with the expected structure."),
        ),
        ui.panel_conditional(
            "input.input_method === 'Create Table'",
            ui.h5("Tasks"),
            ui.input_text("task_name", "Task Name"),
            ui.input_action_button("add_task", "Add Task"),
            ui.hr(),
            ui.h5("Roles"),
            ui.input_text("role_name", "Role Name"),
            ui.input_numeric("hourly_cost", "Hourly Cost", value=0, min=0, step=1),
            ui.input_select("role_tasks", "Associated Tasks", choices=[], multiple=True),
            ui.input_action_button("add_role", "Add Role"),
            ui.hr(),
            ui.input_numeric("num_periods", "Number of Periods", value=12, min=1, max=36, step=1),
            ui.input_action_button("regen_util", "Regenerate Utilization Inputs"),
            ui.input_action_button("save_schedule", "Save Schedule"),
        ),
        ui.hr(),
        ui.h4("Configuration"),
        ui.input_radio_buttons("time_period", "Select Time Period", ["Weeks", "Months"], selected="Months"),
        ui.input_date("project_start", "Project Start Date", value=datetime(2025,1,1)),
        ui.input_numeric("hours_per_day", "Hours per Day", value=9, min=1, max=24),
        ui.input_numeric("work_days_week", "Working Days per Week", value=5, min=1, max=7),
        ui.input_numeric("work_days_month", "Working Days per Month", value=22, min=1, max=31),
        ui.hr(),
        ui.h4("Cumulative Cost Configuration"),
        ui.input_numeric("advance_pct", "Advance Payment (% of Total Cost)", value=0, min=0, max=100, step=1),
        ui.input_numeric("trigger_pct", "Trigger Percentage for Invoicing (%)", value=-50, min=-100, max=100, step=1),
        ui.hr(),
        ui.h4("Discipline Configuration"),
        ui.input_text("new_disc", "Add New Discipline"),
        ui.input_text("new_keys", "Keywords (comma-separated)"),
        ui.input_action_button("add_disc", "Add Discipline"),
        ui.input_select("mod_disc", "Select Discipline to Modify Color", choices=list(DEFAULT_DISCIPLINES.keys())),
        ui.input_text("new_color", "New Color (hex)", value="#0000ff"),
        ui.input_action_button("upd_color", "Update Color"),
        ui.input_select("rem_disc", "Select Discipline to Remove", choices=list(DEFAULT_DISCIPLINES.keys())),
        ui.input_action_button("del_disc", "Remove Discipline"),
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Project Schedule & Cost Visualization (Shiny)"),
            ui.markdown("Use the sidebar to configure inputs and data. Then explore the charts and tables below."),
        ),
    ),
    # Data preview / creation helpers
    ui.navset_pill(
        ui.nav_panel(
            "Data Entry / Preview",
            ui.h5("Current Tasks"),
            ui.output_text_verbatim("tasks_text"),
            ui.h5("Current Roles"),
            ui.output_table("roles_table"),
            ui.panel_conditional(
                "input.input_method === 'Create Table'",
                ui.hr(),
                ui.h5("Enter Utilization (0..1) for each Role-Task by Period"),
                ui.output_ui("utilization_ui"),
            ),
            ui.panel_conditional(
                "input.input_method === 'Upload File'",
                ui.hr(),
                ui.output_table("uploaded_head"),
            ),
        ),
        ui.nav_panel(
            "KPIs",
            ui.output_text_verbatim("kpi_text"),
            ui.h6("Total Man-Hours by Period"),
            ui.output_table("manhours_by_period"),
            ui.h6("Total Man-Hours by Discipline"),
            ui.output_table("manhours_by_disc"),
            ui.h6("Total Man-Hours and Costs by Role"),
            ui.output_table("role_summary"),
        ),
        ui.nav_panel(
            "Charts",
            ui.h5("Man-Hours by Service (Heatmap)"),
            ui.output_plot("heat_service", height="auto"),
            ui.download_button("dl_heat_service", "Download PNG: Man-Hours by Service"),
            ui.hr(),
            ui.h5("Man-Hours by Role (Heatmap)"),
            ui.output_plot("heat_role", height="auto"),
            ui.download_button("dl_heat_role", "Download PNG: Man-Hours by Role"),
            ui.hr(),
            ui.h5("Man-Hours per Discipline Over Time (Stacked)"),
            ui.output_plot("hist_disc"),
            ui.download_button("dl_hist_disc", "Download PNG: Man-Hours Histogram"),
            ui.hr(),
            ui.h5("Donut: Total Man-Hours per Discipline"),
            ui.output_plot("donut_disc"),
            ui.download_button("dl_donut_disc", "Download PNG: Donut"),
            ui.hr(),
            ui.h5("Total Cost by Period"),
            ui.output_plot("cost_by_period_plot"),
            ui.download_button("dl_cost_by_period", "Download PNG: Cost by Period"),
            ui.hr(),
            ui.h5("Total Cost by Task"),
            ui.output_plot("cost_by_task_plot"),
            ui.download_button("dl_cost_by_task", "Download PNG: Cost by Task"),
            ui.hr(),
            ui.h5("Cumulative Cash Flow by Period"),
            ui.output_plot("cum_cash_plot"),
            ui.download_button("dl_cum_cash", "Download PNG: Cumulative Cash Flow"),
        ),
        ui.nav_panel(
            "Tables",
            ui.h5("Total Cost and Man-Hours by Task"),
            ui.output_table("task_summary"),
            ui.hr(),
            ui.h5("Suggested Invoice Schedule"),
            ui.output_table("invoice_table"),
            ui.download_button("dl_invoice_csv", "Download Invoice Schedule CSV"),
        ),
    ),
)

# ------------------------------
# Server
# ------------------------------

def server(input, output, session):
    # Matplotlib style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Reactive state
    tasks = reactive.Value([])  # list[str]
    roles = reactive.Value([])  # list[dict{Role, Hourly Cost, Tasks}]
    schedule_df = reactive.Value(pd.DataFrame())
    disciplines = reactive.Value({k: dict(v) for k, v in DEFAULT_DISCIPLINES.items()})

    # Derived values
    @reactive.Calc
    def currency_symbol() -> str:
        c = (input.currency() or "$").strip()
        return c if c else "$"

    @reactive.Calc
    def hours_per_period() -> float:
        if input.time_period() == "Months":
            return input.hours_per_day() * input.work_days_month()
        return input.hours_per_day() * input.work_days_week()

    # ------------------
    # Data Input helpers
    # ------------------
    @render.text
    def tasks_text():
        t = tasks()
        return ", ".join(t) if t else "—"

    @render.table
    def roles_table():
        r = roles()
        if not r:
            return pd.DataFrame()
        return pd.DataFrame(
            [{"Role": x["Role"], "Hourly Cost": x["Hourly Cost"], "Tasks": ", ".join(x["Tasks"]) } for x in r]
        )

    # Update task list
    @reactive.Effect
    @reactive.event(input.add_task)
    def _add_task():
        name = (input.task_name() or "").strip()
        if not name:
            return
        t = tasks()
        if name not in t:
            tasks.set(t + [name])
            # update role_tasks choices
            session.send_input_message("role_tasks", {"choices": tasks()})
            session.send_input_message("task_name", {"value": ""})

    # Add role
    @reactive.Effect
    @reactive.event(input.add_role)
    def _add_role():
        role_name = (input.role_name() or "").strip()
        if not role_name:
            return
        cost = float(input.hourly_cost() or 0)
        assoc = list(input.role_tasks() or [])
        if not assoc:
            return
        r = roles()
        r.append({"Role": role_name, "Hourly Cost": cost, "Tasks": assoc})
        roles.set(r)
        session.send_input_message("role_name", {"value": ""})
        session.send_input_message("hourly_cost", {"value": 0})
        session.send_input_message("role_tasks", {"selected": []})

    # Disciplines management
    @reactive.Effect
    @reactive.event(input.add_disc)
    def _add_disc():
        nd = (input.new_disc() or "").strip()
        nk = (input.new_keys() or "").strip()
        if nd and nk:
            d = disciplines()
            if nd not in d:
                d[nd] = {"keywords": [k.strip() for k in nk.split(",") if k.strip()], "color": "#%06x" % random.randint(0, 0xFFFFFF)}
                disciplines.set({**d})
                # refresh selects
                session.send_input_message("mod_disc", {"choices": list(disciplines().keys())})
                session.send_input_message("rem_disc", {"choices": list(disciplines().keys())})

    @reactive.Effect
    @reactive.event(input.upd_color)
    def _upd_color():
        sel = input.mod_disc()
        hexcol = (input.new_color() or "").strip()
        if sel and hexcol:
            d = disciplines()
            if sel in d:
                d[sel]["color"] = hexcol
                disciplines.set({**d})

    @reactive.Effect
    @reactive.event(input.del_disc)
    def _del_disc():
        sel = input.rem_disc()
        d = disciplines()
        if sel in d:
            del d[sel]
            disciplines.set({**d})
            session.send_input_message("mod_disc", {"choices": list(disciplines().keys())})
            session.send_input_message("rem_disc", {"choices": list(disciplines().keys())})

    # Template download
    @reactive.Effect
    @reactive.event(input.download_template)
    def _download_template_click():
        # Use a client-side download by sending file content via download handler below
        session.download(
            filename="schedule_template.csv",
            media_type="text/csv",
            data=create_template_csv().encode("utf-8"),
        )

    # Uploaded file head preview
    @render.table
    def uploaded_head():
        f = input.uploaded()
        if not f:
            return pd.DataFrame()
        file = f[0]
        if file["name"].endswith(".xlsx"):
            df = pd.read_excel(file["datapath"], na_values=["", " ", "NA", "NaN"])
        else:
            df = pd.read_csv(file["datapath"], na_values=["", " ", "NA", "NaN"])
        return df.head(10)

    # ------------------------------
    # Utilization dynamic UI (Create Table path)
    # ------------------------------
    def _role_task_pairs() -> List[Tuple[str, str, float]]:
        pairs = []
        for r in roles():
            for t in r["Tasks"]:
                pairs.append((t, r["Role"], float(r["Hourly Cost"])) )
        return pairs

    @render.ui
    def utilization_ui():
        if input.input_method() != "Create Table":
            return ui.div()
        pairs = _role_task_pairs()
        if not pairs:
            return ui.help_text("Add roles with associated tasks first.")
        nper = int(input.num_periods())
        # Build a grid with numeric inputs 0..1 step 0.1 for each period per pair
        rows = []
        header = [ui.tags.th("Service"), ui.tags.th("Role"), ui.tags.th("Hourly Cost")] + [ui.tags.th(str(i)) for i in range(1, nper+1)]
        rows.append(ui.tags.tr(*header))
        for idx, (svc, role_name, cost) in enumerate(pairs):
            cells = [ui.tags.td(svc), ui.tags.td(role_name), ui.tags.td(f"{cost:.2f}")]
            for p in range(1, nper+1):
                input_id = f"util_{idx}_{p}"
                cells.append(ui.tags.td(ui.input_numeric(input_id, "", value=0.0, min=0.0, max=1.0, step=0.1)))
            rows.append(ui.tags.tr(*cells))
        return ui.tags.table({"class": "table table-sm"}, *rows)

    # Save schedule from dynamic inputs
    @reactive.Effect
    @reactive.event(input.save_schedule)
    def _save_schedule():
        if input.input_method() != "Create Table":
            return
        pairs = _role_task_pairs()
        if not pairs:
            return
        nper = int(input.num_periods())
        data = []
        for idx, (svc, role_name, cost) in enumerate(pairs):
            row = {"Service": svc, "Role": role_name, "Hourly Cost": cost}
            for p in range(1, nper+1):
                input_id = f"util_{idx}_{p}"
                val = input.get_value(input_id)
                row[str(p)] = float(val or 0.0)
            data.append(row)
        schedule_df.set(pd.DataFrame(data))

    # If user uploads a file, parse into schedule_df
    @reactive.Effect
    def _load_uploaded():
        if input.input_method() != "Upload File":
            return
        f = input.uploaded()
        if not f:
            return
        file = f[0]
        if file["name"].endswith(".xlsx"):
            df = pd.read_excel(file["datapath"], na_values=["", " ", "NA", "NaN"])
        else:
            df = pd.read_csv(file["datapath"], na_values=["", " ", "NA", "NaN"])
        schedule_df.set(df)

    # ------------------------------
    # Core computations
    # ------------------------------
    @reactive.Calc
    def processed():
        df = schedule_df()
        if df is None or df.empty:
            return None
        df = df.copy()
        df.columns = df.columns.str.strip()
        service_col = "Service"
        role_col = "Role"
        cost_col = "Hourly Cost"
        all_period_columns = df.columns[3:]
        # Clean numeric
        for col in all_period_columns:
            df[col] = (
                df[col].astype(str).str.strip().replace({',': '.', '%': ''}, regex=True).replace('', np.nan)
            )
        df[all_period_columns] = df[all_period_columns].apply(pd.to_numeric, errors='coerce')
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
        # Determine period range actually used
        non_empty_periods = df[all_period_columns].columns[(df[all_period_columns].notna() & (df[all_period_columns] != 0)).any()]
        if len(non_empty_periods) == 0:
            period_columns = list(all_period_columns)
        else:
            nums = [int(c) for c in non_empty_periods]
            mi, ma = min(nums), max(nums)
            period_columns = [str(i) for i in range(mi, ma+1) if str(i) in all_period_columns]
        # Date mapping
        period_col_date_mapping = {}
        for col in period_columns:
            p = int(col)
            start_d, _ = get_period_start_end_dates(p, input.project_start(), input.time_period())
            period_col_date_mapping[col] = start_d
        # Disciplines
        disc_map = disciplines()
        df["Discipline"] = df[role_col].apply(lambda x: map_discipline(x, disc_map))
        # Man-hours & cost
        hpp = hours_per_period()
        total_man_hours = df[period_columns].sum().sum() * hpp
        df["Cost"] = df[period_columns].sum(axis=1) * hpp * df[cost_col]
        total_cost = float(df["Cost"].sum())
        avg_cost = (total_cost / total_man_hours) if total_man_hours > 0 else 0.0
        # Aggregations
        man_hours_by_period = (df[period_columns].sum() * hpp)
        man_hours_by_discipline = df.groupby("Discipline")[period_columns].sum().sum(axis=1) * hpp
        man_hours_by_role = df.groupby(role_col)[period_columns].sum().sum(axis=1) * hpp
        cost_by_role = df.groupby(role_col).apply(lambda x: (x[period_columns].sum(axis=1) * hpp * x[cost_col]).sum(), include_groups=False)
        man_hours_by_task = df.groupby(service_col)[period_columns].sum().sum(axis=1) * hpp
        cost_by_task = df.groupby(service_col).apply(lambda x: (x[period_columns].sum(axis=1) * hpp * x[cost_col]).sum(), include_groups=False)
        # Discipline stacked time series data
        disc_list = sorted(df["Discipline"].unique())
        disc_colors = {d: disc_map.get(d, {}).get("color", "#000000") for d in disc_list}
        disc_time = pd.DataFrame(index=disc_list, columns=period_columns)
        for dsc in disc_list:
            disc_rows = df[df["Discipline"] == dsc]
            for col in period_columns:
                total = 0.0
                for _, row in disc_rows.iterrows():
                    pct = row[col]
                    if pd.notna(pct) and 0 <= pct <= 1:
                        total += pct * hpp
                disc_time.loc[dsc, col] = total
        disc_time = disc_time.apply(pd.to_numeric, errors='coerce').fillna(0)
        disc_time_t = disc_time.transpose()
        disc_time_t.index = [period_col_date_mapping[c] for c in disc_time_t.index]
        disc_time_t.sort_index(inplace=True)
        # Cost by period series
        cost_by_period = pd.Series(0.0, index=period_columns)
        for col in period_columns:
            for _, row in df.iterrows():
                pct = row[col]
                if pd.notna(pct) and 0 <= pct <= 1:
                    mh = pct * hpp
                    cost_by_period[col] += mh * row[cost_col]
        # Cumulative cash flow
        period_costs = pd.Series(0.0, index=period_columns)
        for col in period_columns:
            for _, row in df.iterrows():
                pct = row[col]
                if pd.notna(pct) and 0 <= pct <= 1:
                    mh = pct * hpp
                    period_costs[col] -= mh * row[cost_col]
        advance_payment = total_cost * (float(input.advance_pct()) / 100.0)
        cash_flows = pd.Series(0.0, index=period_columns)
        if period_columns:
            cash_flows[period_columns[0]] += advance_payment
        cumulative = pd.Series(0.0, index=period_columns)
        running = advance_payment
        trigger_threshold = total_cost * (float(input.trigger_pct()) / 100.0)
        target_balance = advance_payment
        total_invoiced = advance_payment
        for col in period_columns:
            running += period_costs[col]
            cumulative[col] = running
            if running <= trigger_threshold and col != period_columns[-1]:
                invoice_amount = target_balance - running
                if total_invoiced + invoice_amount > total_cost:
                    invoice_amount = total_cost - total_invoiced
                if invoice_amount > 0:
                    running += invoice_amount
                    cash_flows[col] += invoice_amount
                    cumulative[col] = running
                    total_invoiced += invoice_amount
        if period_columns and total_invoiced < total_cost:
            last = period_columns[-1]
            final_invoice = total_cost - total_invoiced
            if final_invoice > 0:
                running += final_invoice
                cash_flows[last] += final_invoice
                cumulative[last] = running
                total_invoiced += final_invoice
        cum_df = pd.DataFrame({
            "Period": period_columns,
            "Start Date": [period_col_date_mapping[c] for c in period_columns],
            "Cash Flow": cash_flows,
            "Cumulative Cash Flow": cumulative,
        })
        # Invoice schedule
        inv_rows = []
        inv_no = 1
        for _, row in cum_df.iterrows():
            cf = float(row["Cash Flow"])
            if cf > 0:
                pct = (cf / total_cost * 100.0) if total_cost > 0 else 0.0
                inv_rows.append({
                    "Invoice Number": f"Invoice {inv_no}",
                    "Submission Date": row["Start Date"],
                    "Amount": cf,
                    "Percentage of Total Cost": pct,
                })
                inv_no += 1
        invoice_df = pd.DataFrame(inv_rows)
        # Period date map for plotting x labels
        return dict(
            df=df,
            service_col=service_col,
            role_col=role_col,
            cost_col=cost_col,
            period_columns=period_columns,
            period_dates={c: get_period_start_end_dates(int(c), input.project_start(), input.time_period())[0] for c in period_columns},
            total_man_hours=float(total_man_hours),
            total_cost=float(total_cost),
            avg_cost=float(avg_cost),
            man_hours_by_period=man_hours_by_period,
            man_hours_by_discipline=man_hours_by_discipline,
            man_hours_by_role=man_hours_by_role,
            cost_by_role=cost_by_role,
            disc_time_t=disc_time_t,
            disc_colors={d: disciplines()[d]["color"] for d in sorted(disciplines().keys()) if d in disciplines()},
            cost_by_period=cost_by_period,
            man_hours_by_task=man_hours_by_task,
            cost_by_task=cost_by_task,
            cumulative_df=cum_df,
        )

    # ------------------------------
    # KPIs & Tables
    # ------------------------------
    @render.text
    def kpi_text():
        P = processed()
        if not P:
            return "Upload or create a schedule to see KPIs."
        return (
            f"Total Man-Hours (All Periods): {P['total_man_hours']:.2f}\n"
            f"Total Cost (All Periods): {currency_symbol()}{P['total_cost']:.2f}\n"
            f"Average Hourly Cost: {currency_symbol()}{P['avg_cost']:.2f}"
        )

    @render.table
    def manhours_by_period():
        P = processed()
        if not P:
            return pd.DataFrame()
        df = pd.DataFrame({
            "Period": P["period_columns"],
            "Start Date": [P["period_dates"][c] for c in P["period_columns"]],
            "Man-Hours": P["man_hours_by_period"].values,
        })
        return df

    @render.table
    def manhours_by_disc():
        P = processed()
        if not P:
            return pd.DataFrame()
        return P["man_hours_by_discipline"].reset_index().rename(columns={0: "Man-Hours", "index": "Discipline"})

    @render.table
    def role_summary():
        P = processed()
        if not P:
            return pd.DataFrame()
        df = pd.DataFrame({
            "Role": P["man_hours_by_role"].index,
            "Man-Hours": P["man_hours_by_role"].values,
            "Cost": P["cost_by_role"].values,
        })
        df["Cost"] = df["Cost"].map(lambda x: f"{currency_symbol()}{x:.2f}")
        return df

    @render.table
    def task_summary():
        P = processed()
        if not P:
            return pd.DataFrame()
        man = P["man_hours_by_task"]
        cost = P["cost_by_task"]
        df = pd.DataFrame({"Service": man.index, "Man-Hours": man.values, "Cost": cost.values})
        df["Cost"] = df["Cost"].map(lambda x: f"{currency_symbol()}{x:.2f}")
        return df

    @render.table
    def invoice_table():
        P = processed()
        if not P:
            return pd.DataFrame()
        cum = P["cumulative_df"]
        rows = []
        inv_no = 1
        for _, row in cum.iterrows():
            cf = float(row["Cash Flow"])
            if cf > 0:
                total_cost = P["total_cost"]
                pct = (cf / total_cost * 100.0) if total_cost > 0 else 0.0
                rows.append({
                    "Invoice Number": f"Invoice {inv_no}",
                    "Submission Date": row["Start Date"],
                    "Amount": f"{currency_symbol()}{cf:.2f}",
                    "Percentage of Total Cost": f"{pct:.2f}%",
                })
                inv_no += 1
        return pd.DataFrame(rows)

    # ------------------------------
    # Plots + downloads (each builds its own figure)
    # ------------------------------
    def _date_formatter(ax):
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, fontsize=9)
        ax.grid(True, which='major', linestyle='--', alpha=0.5)

    def _buf_from_fig(fig) -> bytes:
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        b.seek(0)
        return b.getvalue()

    @render.plot
    def heat_service():
        P = processed()
        if not P:
            return
        df = P["df"]
        period_cols = P["period_columns"]
        hpp = hours_per_period()
        # Build pivot
        rows = []
        for task in df[P["service_col"]].unique():
            sub = df[df[P["service_col"]] == task]
            for c in period_cols:
                s = sub[c].sum()
                if pd.notna(s):
                    rows.append({"Task": task, "Period": c, "Man_Hours": s * hpp})
        gdf = pd.DataFrame(rows)
        if gdf.empty:
            return
        pivot = gdf.pivot(index="Task", columns="Period", values="Man_Hours").fillna(0)
        for c in period_cols:
            if c not in pivot.columns:
                pivot[c] = 0.0
        order_map = get_earliest_period(df, P["service_col"], period_cols)
        tasks_ordered = sorted(pivot.index, key=lambda x: order_map.get(x, 10**9))
        periods_ordered = sorted([c for c in period_cols], key=int)
        pivot = pivot.loc[tasks_ordered, periods_ordered]
        # Width heuristic
        fig, ax = plt.subplots(figsize=(max(10, len(periods_ordered) * 0.6), max(6, len(tasks_ordered) * 0.5)))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("StrongBlueGreen", ["#ffffff", "#a6cee3", "#006d2c"])
        sns.heatmap(
            pivot,
            cmap=cmap,
            annot=pivot.applymap(lambda v: "" if (isinstance(v, (int,float)) and v == 0) else f"{v:.1f}"),
            fmt="",
            cbar_kws={"label": "Man-Hours"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 8, "color": "black"},
            vmin=0,
            vmax=max(pivot.values.max() * 1.5 or 1, 15),
        )
        ax.set_xticklabels([P["period_dates"][p].strftime("%Y-%m-%d") for p in periods_ordered], rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(tasks_ordered, rotation=0, fontsize=10)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel("Tasks")
        ax.set_title(f"Man-Hours by Service ({input.time_period()})")
        plt.tight_layout(pad=2.0)
        return fig

    @render.download(filename="man_hours_by_service.png")
    def dl_heat_service():
        fig = heat_service.instance().value
        if fig is None:
            return b""
        return _buf_from_fig(fig)

    @render.plot
    def heat_role():
        P = processed()
        if not P:
            return
        df = P["df"]
        period_cols = P["period_columns"]
        hpp = hours_per_period()
        rows = []
        for role in df[P["role_col"]].unique():
            sub = df[df[P["role_col"]] == role]
            for c in period_cols:
                s = sub[c].sum()
                if pd.notna(s):
                    rows.append({"Role": role, "Period": c, "Man_Hours": s * hpp})
        gdf = pd.DataFrame(rows)
        if gdf.empty:
            return
        pivot = gdf.pivot(index="Role", columns="Period", values="Man_Hours").fillna(0)
        for c in period_cols:
            if c not in pivot.columns:
                pivot[c] = 0.0
        order_map = get_earliest_period(df, P["role_col"], period_cols)
        roles_ordered = sorted(pivot.index, key=lambda x: order_map.get(x, 10**9))
        periods_ordered = sorted([c for c in period_cols], key=int)
        pivot = pivot.loc[roles_ordered, periods_ordered]
        fig, ax = plt.subplots(figsize=(max(10, len(periods_ordered) * 0.7), max(6, len(roles_ordered) * 0.5)))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("StrongYellowRed", ["#ffffff", "#f7e8aa", "#d62728"])
        sns.heatmap(
            pivot,
            cmap=cmap,
            annot=pivot.applymap(lambda v: "" if (isinstance(v, (int,float)) and v == 0) else f"{v:.1f}"),
            fmt="",
            cbar_kws={"label": "Man-Hours"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 8, "color": "black"},
            vmin=0,
            vmax=max(pivot.values.max() * 1.5 or 1, 15),
        )
        ax.set_xticklabels([P["period_dates"][p].strftime("%Y-%m-%d") for p in periods_ordered], rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(roles_ordered, rotation=0, fontsize=10)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel("Roles")
        ax.set_title(f"Man-Hours by Role ({input.time_period()})")
        plt.tight_layout(pad=2.0)
        return fig

    @render.download(filename="man_hours_by_role.png")
    def dl_heat_role():
        fig = heat_role.instance().value
        if fig is None:
            return b""
        return _buf_from_fig(fig)

    @render.plot
    def hist_disc():
        P = processed()
        if not P:
            return
        dt = P["disc_time_t"]
        if dt.empty:
            return
        disc_cols = list(dt.columns)
        colors = {d: disciplines().get(d, {}).get("color", "#000000") for d in disc_cols}
        fig, ax = plt.subplots(figsize=(14, 10))
        bottom = np.zeros(len(dt))
        bar_width = 15 if input.time_period() == "Months" else 2
        for d in disc_cols:
            vals = dt[d].values.astype(float)
            bars = ax.bar(dt.index, vals, bottom=bottom, label=d, color=colors.get(d, "#000000"), width=bar_width, alpha=0.85)
            threshold = 10 if input.time_period() == "Months" else 2
            for b in bars:
                h = b.get_height()
                if h > threshold:
                    ax.text(b.get_x() + b.get_width()/2, b.get_y() + h/2, f"{h:.0f}", ha="center", va="center", fontsize=8, rotation=90, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            bottom += vals
        _date_formatter(ax)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel("Man-Hours")
        ax.set_title(f"Man-Hours per Discipline Over {input.time_period()}")
        ax.legend(title="Discipline", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        plt.tight_layout(pad=2.0)
        return fig

    @render.download(filename="man_hours_histogram.png")
    def dl_hist_disc():
        fig = hist_disc.instance().value
        if fig is None:
            return b""
        return _buf_from_fig(fig)

    @render.plot
    def donut_disc():
        P = processed()
        if not P:
            return
        dt = P["disc_time_t"]
        if dt.empty:
            return
        totals = dt.sum(axis=0)
        labels = totals.index
        sizes = (totals / totals.sum() * 100.0).values if totals.sum() != 0 else np.zeros(len(totals))
        colors = [disciplines().get(d, {}).get("color", "#000000") for d in labels]
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            autopct="%1.1f%%",
            pctdistance=0.85,
            labeldistance=1.05,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"color": "black", "fontsize": 12},
        )
        center = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(center)
        ax.axis("equal")
        ax.set_title("Percentage Distribution of Total Man-Hours per Discipline")
        plt.tight_layout()
        return fig

    @render.download(filename="donut_chart.png")
    def dl_donut_disc():
        fig = donut_disc.instance().value
        if fig is None:
            return b""
        return _buf_from_fig(fig)

    @render.plot
    def cost_by_period_plot():
        P = processed()
        if not P:
            return
        df = pd.DataFrame({
            "Period": P["period_columns"],
            "Start Date": [P["period_dates"][c] for c in P["period_columns"]],
            "Cost": P["cost_by_period"].values,
        })
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 15 if input.time_period() == "Months" else 2
        bars = ax.bar(df["Start Date"], df["Cost"].astype(float), width=bar_width)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h, f"{currency_symbol()}{h:.0f}", ha="center", va="bottom", fontsize=8)
        _date_formatter(ax)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel(f"Cost ({currency_symbol()})")
        ax.set_title(f"Total Cost by {input.time_period()}")
        plt.tight_layout()
        return fig

    @render.download(filename="cost_by_period.png")
    def dl_cost_by_period():
        fig = cost_by_period_plot.instance().value
        if fig is None:
            return b""
        return _buf_from_fig(fig)

    @render.plot
    def cost_by_task_plot():
        P = processed()
        if not P:
            return
        cost_by_task = P["cost_by_task"].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(cost_by_task.index, cost_by_task.values, color="#2ecc71")
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h, f"{currency_symbol()}{h:.0f}", ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(f"Cost ({currency_symbol()})")
        ax.set_title("Total Cost by Task")
        plt.tight_layout()
        return fig

    @render.download(filename="cost_by_task.png")
    def dl_cost_by_task():
        fig = cost_by_task_plot.instance().value
        if fig is None:
            return b""
        return _buf_from_fig(fig)

    @render.plot
    def cum_cash_plot():
        P = processed()
        if not P:
            return
        cum = P["cumulative_df"].copy()
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 15 if input.time_period() == "Months" else 2
        # bars: negative costs (red), positive (green)
        for _, row in cum.iterrows():
            cf = float(row["Cash Flow"])
            color = "#ff0000" if cf < 0 else "#008000"
            bars = ax.bar(row["Start Date"], cf, width=bar_width, color=color)
            if abs(cf) > 0:
                ax.text(bars[0].get_x() + bars[0].get_width()/2, cf + (100 if cf > 0 else -100), f"{currency_symbol()}{cf:.0f}", ha="center", va="bottom" if cf > 0 else "top", fontsize=8)
        # cumulative line
        ax.plot(cum["Start Date"], cum["Cumulative Cash Flow"].astype(float), linewidth=2, marker="o", label="Cumulative Cash Flow")
        _date_formatter(ax)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel(f"Cash Flow ({currency_symbol()})")
        ax.set_title(f"Cumulative Cash Flow by {input.time_period()}")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.legend()
        plt.tight_layout()
        return fig

    @render.download(filename="cumulative_cash_flow.png")
    def dl_cum_cash():
        fig = cum_cash_plot.instance().value
        if fig is None:
            return b""
        return _buf_from_fig(fig)

    # Invoice CSV download
    @render.download(filename="invoice_schedule.csv")
    def dl_invoice_csv():
        P = processed()
        if not P:
            return b""
        cum = P["cumulative_df"]
        rows = []
        inv_no = 1
        for _, row in cum.iterrows():
            cf = float(row["Cash Flow"])
            if cf > 0:
                pct = (cf / P["total_cost"] * 100.0) if P["total_cost"] > 0 else 0.0
                rows.append({
                    "Invoice Number": f"Invoice {inv_no}",
                    "Submission Date": row["Start Date"].strftime("%Y-%m-%d"),
                    "Amount": f"{currency_symbol()}{cf:.2f}",
                    "Percentage of Total Cost": f"{pct:.2f}%",
                })
                inv_no += 1
        csv = pd.DataFrame(rows).to_csv(index=False)
        return csv.encode("utf-8")


app = App(app_ui, server)
