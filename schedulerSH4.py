# app.py — Shiny (Python) Project Scheduler
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
# Utility functions
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
# Default disciplines with standard colors
# ------------------------------
DEFAULT_DISCIPLINES = {
    "Civil": {"keywords": ["Civil", "GIS", "Geotechnical"], "color": "#1f77b4"},  # blue
    "Electrical": {"keywords": ["Electrical", "PV"], "color": "#ff7f0e"},  # orange
    "HSE": {"keywords": ["HSE", "Safety"], "color": "#2ca02c"},  # green
    "Instrument": {"keywords": ["Instrument", "SCADA", "I&C", "Automation"], "color": "#d62728"},  # red
    "Management": {"keywords": ["Project Manager", "DCC", "Project Engineer", "Manager"], "color": "#9467bd"},  # purple
    "Mechanical": {"keywords": ["Mechanical"], "color": "#8c564b"},  # brown
    "Site Supervision": {"keywords": ["Supervisor", "Commissioning"], "color": "#e377c2"},  # pink
    "Other": {"keywords": [], "color": "#7f7f7f"},  # gray
}

# ------------------------------
# Shiny UI with larger sidebar
# ------------------------------
app_ui = ui.TagList(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.1/dist/flatly/bootstrap.min.css"),
        ui.tags.style("""
            .sidebar {
                width: 350px !important;
                padding: 20px !important;
            }
            .form-control {
                font-size: 14px !important;
            }
            .form-group {
                margin-bottom: 15px !important;
            }
            input[type=number] {
                width: 80px !important;
                font-size: 14px !important;
            }
            .table { 
                font-size: 12px; 
            }
            .card { 
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
                margin-bottom: 20px;
            }
            .nav-pills .nav-link { 
                font-size: 14px; 
            }
            body { 
                background-color: #f8f9fa; 
            }
        """)
    ),
    ui.page_sidebar(
        ui.sidebar(
            ui.h4("Data Input Method"),
            ui.input_radio_buttons(
                "input_method", "Choose Input Method", ["Create Table", "Upload File"], selected="Create Table"
            ),
            ui.input_text("currency", "Enter Currency Symbol", value="$"),
            ui.panel_conditional(
                "input.input_method === 'Upload File'",
                ui.input_file("uploaded", "Upload Schedule CSV or Excel", multiple=False, accept=[".csv", ".xlsx"]),
                ui.download_button("download_template", "Download CSV Template"),
                ui.help_text("Click the button to download a CSV with the expected structure."),
            ),
            ui.panel_conditional(
                "input.input_method === 'Create Table'",
                ui.h5("Tasks"),
                ui.layout_columns(
                    ui.input_text("task_name", "Task Name", width="100%"),
                    ui.input_action_button("add_task", "Add Task", class_="btn-primary"),
                    col_widths=[8, 4]
                ),
                ui.output_ui("current_tasks_list"),
                ui.hr(),
                ui.h5("Roles"),
                ui.layout_columns(
                    ui.input_text("role_name", "Role Name", width="100%"),
                    ui.input_numeric("hourly_cost", "Hourly Cost", value=0, min=0, step=1, width="100%"),
                    col_widths=[6, 6]
                ),
                ui.input_selectize("role_tasks", "Associated Tasks", choices=[], multiple=True),
                ui.input_action_button("add_role", "Add Role", class_="btn-primary"),
                ui.output_ui("current_roles_list"),
                ui.hr(),
                ui.input_numeric("num_periods", "Number of Periods", value=12, min=1, max=36, step=1),
                ui.input_action_button("regen_util", "Regenerate Utilization Inputs", class_="btn-secondary"),
                ui.input_action_button("save_schedule", "Save Schedule", class_="btn-primary"),
            ),
            ui.hr(),
            ui.h4("Configuration"),
            ui.input_radio_buttons("time_period", "Select Time Period", ["Weeks", "Months"], selected="Months"),
            ui.input_date("project_start", "Project Start Date", value=datetime(2025,1,1)),
            ui.layout_columns(
                ui.input_numeric("hours_per_day", "Hours per Day", value=9, min=1, max=24),
                ui.input_numeric("work_days_week", "Working Days/Week", value=5, min=1, max=7),
                ui.input_numeric("work_days_month", "Working Days/Month", value=22, min=1, max=31),
                col_widths=[4, 4, 4]
            ),
            ui.hr(),
            ui.h4("Cumulative Cost Configuration"),
            ui.layout_columns(
                ui.input_numeric("advance_pct", "Advance Payment (%)", value=0, min=0, max=100, step=1),
                ui.input_numeric("trigger_pct", "Trigger Percentage (%)", value=-50, min=-100, max=100, step=1),
                col_widths=[6, 6]
            ),
            ui.hr(),
            ui.h4("Discipline Configuration"),
            ui.layout_columns(
                ui.input_text("new_disc", "New Discipline"),
                ui.input_text("new_keys", "Keywords (comma-sep)"),
                col_widths=[6, 6]
            ),
            ui.input_action_button("add_disc", "Add Discipline", class_="btn-primary"),
            ui.layout_columns(
                ui.input_select("mod_disc", "Modify Color", choices=list(DEFAULT_DISCIPLINES.keys())),
                ui.input_text("new_color", "Hex Color", value="#0000ff"),
                ui.input_action_button("upd_color", "Update", class_="btn-secondary"),
                col_widths=[4, 4, 4]
            ),
            ui.layout_columns(
                ui.input_select("rem_disc", "Remove Discipline", choices=list(DEFAULT_DISCIPLINES.keys())),
                ui.input_action_button("del_disc", "Remove", class_="btn-danger"),
                col_widths=[8, 4]
            ),
            width=350
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Project Schedule & Cost Visualization"),
                ui.markdown("Use the sidebar to configure inputs and data. Then explore the charts and tables below."),
                full_screen=True
            ),
        ),
        ui.navset_pill(
            ui.nav_panel(
                "Data Entry / Preview",
                ui.h5("Current Tasks"),
                ui.output_ui("tasks_text"),
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
                ui.hr(),
                ui.h5("Current Schedule Preview"),
                ui.output_table("schedule_preview"),
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
                ui.output_plot("heat_service", height="800px"),
                ui.hr(),
                ui.h5("Man-Hours by Role (Heatmap)"),
                ui.output_plot("heat_role", height="800px"),
                ui.hr(),
                ui.h5("Man-Hours per Discipline Over Time (Stacked)"),
                ui.output_plot("hist_disc", height="600px"),
                ui.hr(),
                ui.h5("Donut: Total Man-Hours per Discipline"),
                ui.output_plot("donut_disc", height="600px"),
                ui.hr(),
                ui.h5("Total Cost by Period"),
                ui.output_plot("cost_by_period_plot", height="500px"),
                ui.hr(),
                ui.h5("Total Cost by Task"),
                ui.output_plot("cost_by_task_plot", height="500px"),
                ui.hr(),
                ui.h5("Cumulative Cash Flow by Period"),
                ui.output_plot("cum_cash_plot", height="500px"),
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
    # Data Input helpers - rewritten create table section
    # ------------------
    @render.ui
    def tasks_text():
        t = tasks()
        if not t:
            return ui.div("No tasks added yet.", class_="text-muted")
        return ui.div(
            ui.ul(*[ui.li(task) for task in t]),
            style="max-height: 200px; overflow-y: auto;"
        )

    @render.ui
    def current_tasks_list():
        t = tasks()
        if not t:
            return ui.div()
        return ui.div(
            ui.markdown("**Current Tasks:**"),
            ui.ul(*[ui.li(task) for task in t]),
            style="margin-bottom: 15px; max-height: 150px; overflow-y: auto;"
        )

    @render.ui
    def current_roles_list():
        r = roles()
        if not r:
            return ui.div()
        return ui.div(
            ui.markdown("**Current Roles:**"),
            ui.ul(*[
                ui.li(f"{role['Role']} (${role['Hourly Cost']}/h) - Tasks: {', '.join(role['Tasks'])}")
                for role in r
            ]),
            style="margin-bottom: 15px; max-height: 150px; overflow-y: auto;"
        )

    @render.table
    def roles_table():
        r = roles()
        if not r:
            return pd.DataFrame(columns=["Role", "Hourly Cost", "Tasks"])
        return pd.DataFrame(
            [{"Role": x["Role"], "Hourly Cost": x["Hourly Cost"], "Tasks": ", ".join(x["Tasks"]) } for x in r]
        )

    # Update task list
    @reactive.Effect
    @reactive.event(input.add_task)
    def _add_task():
        name = (input.task_name() or "").strip()
        if not name:
            ui.notification_show("Please enter a task name", duration=3, type="warning")
            return
        t = tasks()
        if name not in t:
            t.append(name)
            tasks.set(t)
            ui.update_select("role_tasks", choices=tasks())
            ui.update_text("task_name", value="")
            ui.notification_show(f"Task '{name}' added", duration=3, type="message")

    # Add role - make unique by name, update tasks and cost if exists
    @reactive.Effect
    @reactive.event(input.add_role)
    def _add_role():
        role_name = (input.role_name() or "").strip()
        if not role_name:
            ui.notification_show("Please enter a role name", duration=3, type="warning")
            return
        cost = float(input.hourly_cost() or 0)
        if cost <= 0:
            ui.notification_show("Hourly cost must be positive", duration=3, type="warning")
            return
        assoc = list(input.role_tasks() or [])
        if not assoc:
            ui.notification_show("Please select at least one task", duration=3, type="warning")
            return
            
        r = roles()
        existing = next((x for x in r if x["Role"] == role_name), None)
        if existing:
            existing["Hourly Cost"] = cost
            existing["Tasks"] = list(set(existing["Tasks"] + assoc))
            ui.notification_show(f"Role '{role_name}' updated", duration=3, type="message")
        else:
            r.append({"Role": role_name, "Hourly Cost": cost, "Tasks": assoc})
            ui.notification_show(f"Role '{role_name}' added", duration=3, type="message")
        roles.set(r)
        ui.update_text("role_name", value="")
        ui.update_numeric("hourly_cost", value=0)
        ui.update_select("role_tasks", selected=[])

    # Disciplines management
    @reactive.Effect
    @reactive.event(input.add_disc)
    def _add_disc():
        nd = (input.new_disc() or "").strip()
        nk = (input.new_keys() or "").strip()
        if not nd:
            ui.notification_show("Please enter a discipline name", duration=3, type="warning")
            return
        if not nk:
            ui.notification_show("Please enter keywords", duration=3, type="warning")
            return
            
        d = disciplines()
        if nd in d:
            ui.notification_show(f"Discipline '{nd}' already exists", duration=3, type="warning")
            return
            
        d[nd] = {
            "keywords": [k.strip() for k in nk.split(",") if k.strip()],
            "color": "#%06x" % random.randint(0, 0xFFFFFF)
        }
        disciplines.set(d)
        ui.update_select("mod_disc", choices=list(d.keys()))
        ui.update_select("rem_disc", choices=list(d.keys()))
        ui.notification_show(f"Discipline '{nd}' added", duration=3, type="message")

    @reactive.Effect
    @reactive.event(input.upd_color)
    def _upd_color():
        sel = input.mod_disc()
        hexcol = (input.new_color() or "").strip()
        if not sel:
            ui.notification_show("Please select a discipline", duration=3, type="warning")
            return
        if not hexcol.startswith("#") or len(hexcol) != 7:
            ui.notification_show("Please enter a valid hex color (e.g. #ff0000)", duration=3, type="warning")
            return
            
        d = disciplines()
        if sel in d:
            d[sel]["color"] = hexcol
            disciplines.set(d)
            ui.notification_show(f"Color updated for '{sel}'", duration=3, type="message")

    @reactive.Effect
    @reactive.event(input.del_disc)
    def _del_disc():
        sel = input.rem_disc()
        if not sel:
            ui.notification_show("Please select a discipline", duration=3, type="warning")
            return
            
        d = disciplines()
        if sel in d:
            del d[sel]
            disciplines.set(d)
            ui.update_select("mod_disc", choices=list(d.keys()))
            ui.update_select("rem_disc", choices=list(d.keys()))
            ui.notification_show(f"Discipline '{sel}' removed", duration=3, type="message")

    # Template download
    @render.download(filename="schedule_template.csv")
    def download_template():
        return create_template_csv()

    # Uploaded file head preview
    @render.table
    def uploaded_head():
        f = input.uploaded()
        if not f:
            return pd.DataFrame(columns=["No file uploaded"])
        file = f[0]
        try:
            if file["name"].endswith(".xlsx"):
                df = pd.read_excel(file["datapath"], na_values=["", " ", "NA", "NaN"])
            else:
                df = pd.read_csv(file["datapath"], na_values=["", " ", "NA", "NaN"])
            return df.head(10)
        except Exception as e:
            return pd.DataFrame({"Error": [f"Could not read file: {str(e)}"]})

    # Schedule preview
    @render.table
    def schedule_preview():
        df = schedule_df()
        if df.empty:
            return pd.DataFrame(columns=["No schedule data available"])
        return df.head(10)

    # ------------------------------
    # Utilization dynamic UI (Create Table path) - rewritten
    # ------------------------------
    def _role_task_pairs() -> List[Tuple[str, str, float]]:
        pairs = []
        for r in roles():
            for t in r["Tasks"]:
                pairs.append((t, r["Role"], float(r["Hourly Cost"])))
        return pairs

    @render.ui
    def utilization_ui():
        if input.input_method() != "Create Table":
            return ui.div()
            
        pairs = _role_task_pairs()
        if not pairs:
            return ui.div(
                ui.markdown("**No role-task pairs defined.**"),
                "Add roles with associated tasks first.",
                class_="text-muted"
            )
            
        nper = int(input.num_periods())
        if nper <= 0:
            return ui.div("Number of periods must be positive", class_="text-danger")
            
        # Build a grid with numeric inputs 0..1 step 0.1 for each period per pair
        rows = []
        header = [
            ui.tags.th("Service"), 
            ui.tags.th("Role"), 
            ui.tags.th("Hourly Cost")
        ] + [ui.tags.th(str(i), style="min-width: 50px;") for i in range(1, nper+1)]
        rows.append(ui.tags.tr(*header))
        
        for idx, (svc, role_name, cost) in enumerate(pairs):
            cells = [
                ui.tags.td(svc), 
                ui.tags.td(role_name), 
                ui.tags.td(f"{cost:.2f}")
            ]
            for p in range(1, nper+1):
                input_id = f"util_{idx}_{p}"
                cells.append(ui.tags.td(
                    ui.input_numeric(
                        input_id, 
                        None, 
                        value=0.0, 
                        min=0.0, 
                        max=1.0, 
                        step=0.1,
                        width="100%"
                    ),
                    style="padding: 2px;"
                ))
            rows.append(ui.tags.tr(*cells))
            
        return ui.tags.div(
            {"style": "overflow-x: auto; max-height: 600px; overflow-y: auto;"},
            ui.tags.table(
                {"class": "table table-sm table-striped table-bordered"}, 
                *rows
            )
        )

    # Regenerate utilization (randomize values)
    @reactive.Effect
    @reactive.event(input.regen_util)
    def _regen_util_values():
        pairs = _role_task_pairs()
        if not pairs:
            ui.notification_show("No role-task pairs defined", duration=3, type="warning")
            return
            
        nper = int(input.num_periods())
        if nper <= 0:
            ui.notification_show("Number of periods must be positive", duration=3, type="warning")
            return
            
        for idx in range(len(pairs)):
            for p in range(1, nper + 1):
                input_id = f"util_{idx}_{p}"
                ui.update_numeric(input_id, value=round(random.uniform(0, 0.9), 1))
        ui.notification_show("Utilization values regenerated", duration=3, type="message")

    # Save schedule from dynamic inputs
    @reactive.Effect
    @reactive.event(input.save_schedule)
    def _save_schedule():
        if input.input_method() != "Create Table":
            return
            
        pairs = _role_task_pairs()
        if not pairs:
            ui.notification_show("No role-task pairs defined", duration=3, type="warning")
            return
            
        nper = int(input.num_periods())
        if nper <= 0:
            ui.notification_show("Number of periods must be positive", duration=3, type="warning")
            return
            
        data = []
        for idx, (svc, role_name, cost) in enumerate(pairs):
            row = {"Service": svc, "Role": role_name, "Hourly Cost": cost}
            for p in range(1, nper+1):
                input_id = f"util_{idx}_{p}"
                val = input[input_id]()
                row[str(p)] = float(val or 0.0)
            data.append(row)
            
        schedule_df.set(pd.DataFrame(data))
        ui.notification_show("Schedule saved successfully", duration=3, type="message")

    # If user uploads a file, parse into schedule_df
    @reactive.Effect
    def _load_uploaded():
        if input.input_method() != "Upload File":
            return
            
        f = input.uploaded()
        if not f:
            return
            
        file = f[0]
        try:
            if file["name"].endswith(".xlsx"):
                df = pd.read_excel(file["datapath"], na_values=["", " ", "NA", "NaN"])
            else:
                df = pd.read_csv(file["datapath"], na_values=["", " ", "NA", "NaN"])
            schedule_df.set(df)
            ui.notification_show("File uploaded successfully", duration=3, type="message")
        except Exception as e:
            ui.notification_show(f"Error reading file: {str(e)}", duration=5, type="error")

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
        all_period_columns = [col for col in df.columns if col.isdigit()]
        
        # Clean numeric
        for col in all_period_columns:
            df[col] = (
                df[col].astype(str).str.strip()
                .replace({',': '.', '%': ''}, regex=True)
                .replace('', np.nan)
            )
        df[all_period_columns] = df[all_period_columns].apply(pd.to_numeric, errors='coerce')
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
        
        # Determine period range actually used
        non_empty_periods = df[all_period_columns].columns[
            (df[all_period_columns].notna() & (df[all_period_columns] != 0)).any()]
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
        cost_by_role = df.groupby(role_col).apply(
            lambda x: (x[period_columns].sum(axis=1) * hpp * x[cost_col]).sum(), 
            include_groups=False
        )
        man_hours_by_task = df.groupby(service_col)[period_columns].sum().sum(axis=1) * hpp
        cost_by_task = df.groupby(service_col).apply(
            lambda x: (x[period_columns].sum(axis=1) * hpp * x[cost_col]).sum(), 
            include_groups=False
        )
        
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
                    period_costs[col] += -mh * row[cost_col]
                    
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
            disc_colors=disc_colors,
            cost_by_period=cost_by_period,
            man_hours_by_task=man_hours_by_task,
            cost_by_task=cost_by_task,
            cumulative_df=cum_df,
            period_costs=period_costs,
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
            return pd.DataFrame(columns=["No data available"])
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
            return pd.DataFrame(columns=["No data available"])
        return P["man_hours_by_discipline"].reset_index().rename(
            columns={0: "Man-Hours", "index": "Discipline"}
        )

    @render.table
    def role_summary():
        P = processed()
        if not P:
            return pd.DataFrame(columns=["No data available"])
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
            return pd.DataFrame(columns=["No data available"])
        man = P["man_hours_by_task"]
        cost = P["cost_by_task"]
        df = pd.DataFrame({
            "Service": man.index, 
            "Man-Hours": man.values, 
            "Cost": cost.values
        })
        df["Cost"] = df["Cost"].map(lambda x: f"{currency_symbol()}{x:.2f}")
        return df

    @render.table
    def invoice_table():
        P = processed()
        if not P:
            return pd.DataFrame(columns=["No data available"])
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
    # Plots with improved styling and visibility
    # ------------------------------
    def _date_formatter(ax):
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, fontsize=9, ha="right")
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        plt.tight_layout()

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
        
        # Adjusted figsize for better visibility
        num_periods = len(periods_ordered)
        num_tasks = len(tasks_ordered)
        fig, ax = plt.subplots(figsize=(max(12, num_periods * 1.0), max(8, num_tasks * 1.0)))
        
        # White to red color gradient
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("blue_white", ["white", "#43a0f7"])
        annot = pivot.applymap(lambda v: "" if v == 0 else f"{v:.0f}")
        
        sns.heatmap(
            pivot,
            cmap=cmap,
            annot=annot,
            fmt="",
            cbar_kws={"label": "Man-Hours"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 8, "color": "black"},
            vmin=0,
            vmax=max(pivot.values.max() * 1.2 or 1, 20),
        )
        
        ax.set_xticklabels(
            [P["period_dates"][p].strftime("%Y-%m-%d") for p in periods_ordered], 
            rotation=45, 
            ha="right", 
            fontsize=9
        )
        ax.set_yticklabels(tasks_ordered, rotation=0, fontsize=9)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel("Tasks")
        ax.set_title(f"Man-Hours by Service ({input.time_period()})")
        max_task_len = max((len(s) for s in tasks_ordered), default=10)
        left = min(0.62, 0.12 + 0.013 * max_task_len)   # more space on the LEFT
        fig.subplots_adjust(left=left, bottom=0.18, right=0.98, top=0.92)
        return fig

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
        
        # Adjusted figsize
        num_periods = len(periods_ordered)
        num_roles = len(roles_ordered)
        fig, ax = plt.subplots(figsize=(max(12, num_periods * 1.0), max(8, num_roles * 1.0)))
        
        # White to red color gradient
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("orange_white", ["white", "salmon"])
        annot = pivot.applymap(lambda v: "" if v == 0 else f"{v:.0f}")
        
        sns.heatmap(
            pivot,
            cmap=cmap,
            annot=annot,
            fmt="",
            cbar_kws={"label": "Man-Hours"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"size": 8, "color": "black"},
            vmin=0,
            vmax=max(pivot.values.max() * 1.2 or 1, 20),
        )
        
        ax.set_xticklabels(
            [P["period_dates"][p].strftime("%Y-%m-%d") for p in periods_ordered], 
            rotation=45, 
            ha="right", 
            fontsize=9
        )
        ax.set_yticklabels(roles_ordered, rotation=0, fontsize=9)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel("Roles")
        ax.set_title(f"Man-Hours by Role ({input.time_period()})")
        max_role_len = max((len(r) for r in roles_ordered), default=10)
        left = min(0.65, 0.12 + 0.013 * max_role_len)   # more LEFT space for long role names
        bottom = min(0.40, 0.16 + 0.005 * len(periods_ordered))  # more BOTTOM for rotated dates
        fig.subplots_adjust(left=left, bottom=bottom, right=0.98, top=0.92)  # extra TOP too
        return fig

    @render.plot
    def hist_disc():
        P = processed()
        if not P:
            return
        dt = P["disc_time_t"]
        if dt.empty:
            return
            
        disc_cols = list(dt.columns)
        # Use discipline colors from config
        palette = [P["disc_colors"][d] for d in disc_cols]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        bottom = np.zeros(len(dt))
        bar_width = 15 if input.time_period() == "Months" else 2
        
        for i, d in enumerate(disc_cols):
            vals = dt[d].values.astype(float)
            bars = ax.bar(
                dt.index, 
                vals, 
                bottom=bottom, 
                label=d, 
                color=palette[i], 
                width=bar_width, 
                alpha=0.85
            )
            threshold = 10 if input.time_period() == "Months" else 2
            for b in bars:
                h = b.get_height()
                if h > threshold:
                    ax.text(
                        b.get_x() + b.get_width()/2, 
                        b.get_y() + h/2, 
                        f"{h:.0f}", 
                        ha="center", 
                        va="center", 
                        fontsize=8, 
                        rotation=90, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                    )
            bottom += vals
            
        _date_formatter(ax)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel("Man-Hours")
        ax.set_title(f"Man-Hours per Discipline Over {input.time_period()}")
        ax.legend(
            title="Discipline", 
            bbox_to_anchor=(1.02, 1), 
            loc="upper left", 
            fontsize=9
        )
        plt.tight_layout(pad=2.0)
        return fig

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
        
        # Use discipline colors from config
        colors = [P["disc_colors"][d] for d in labels]
        
        # Larger figure size to prevent overlapping
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create pie with larger radius and adjust text positions
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
            pctdistance=0.85,
            labeldistance=1.05,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"color": "black", "fontsize": 12},
            radius=1.2
        )
        
        # Add manual annotations for small slices
        for i, (wedge, pct) in enumerate(zip(wedges, sizes)):
            if pct < 5:
                ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
                x = 1.3 * np.cos(np.deg2rad(ang))
                y = 1.3 * np.sin(np.deg2rad(ang))
                ax.annotate(
                    f"{pct:.1f}%", 
                    (x, y), 
                    ha="center", 
                    va="center", 
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
        
        center = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(center)
        ax.axis("equal")
        ax.set_title("Percentage Distribution of Total Man-Hours per Discipline")
        plt.tight_layout()
        return fig

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
        bars = ax.bar(
            df["Start Date"], 
            df["Cost"].astype(float), 
            width=bar_width, 
            color="#2c7fb8"
        )
        
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(
                    b.get_x() + b.get_width()/2, 
                    h, 
                    f"{currency_symbol()}{h:.0f}", 
                    ha="center", 
                    va="bottom", 
                    fontsize=8
                )
                
        _date_formatter(ax)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel(f"Cost ({currency_symbol()})")
        ax.set_title(f"Total Cost by {input.time_period()}")
        plt.tight_layout()
        return fig

    @render.plot
    def cost_by_task_plot():
        P = processed()
        if not P:
            return
        cost_by_task = P["cost_by_task"].sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(
            cost_by_task.index, 
            cost_by_task.values, 
            color="#2c7fb8"
        )
        
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(
                    b.get_x() + b.get_width()/2, 
                    h, 
                    f"{currency_symbol()}{h:.0f}", 
                    ha="center", 
                    va="bottom", 
                    fontsize=8
                )
                
        plt.xticks(rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(f"Cost ({currency_symbol()})")
        ax.set_title("Total Cost by Task")
        plt.tight_layout()
        return fig

    @render.plot
    def cum_cash_plot():
        P = processed()
        if not P:
            return
        cum = P["cumulative_df"].copy()
        period_costs = P["period_costs"]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 15 if input.time_period() == "Months" else 2
        
        # plot negative costs (dark blue)
        ax.bar(
            cum["Start Date"], 
            period_costs.astype(float), 
            width=bar_width, 
            color="#ff7f0e", 
            label="Costs"
        )
        
        # plot positive inflows (orange as requested)
        ax.bar(
            cum["Start Date"], 
            cum["Cash Flow"].astype(float), 
            width=bar_width, 
            color="#43a0f7", 
            label="Inflows"
        )
        
        # add text for costs
        for i, cost in enumerate(period_costs):
            if cost < 0:
                ax.text(
                    cum["Start Date"][i], 
                    cost, 
                    f"{currency_symbol()}{abs(cost):.0f}", 
                    ha="center", 
                    va="top", 
                    fontsize=8
                )
                
        # add text for inflows
        for i, inflow in enumerate(cum["Cash Flow"]):
            if inflow > 0:
                ax.text(
                    cum["Start Date"][i], 
                    inflow, 
                    f"{currency_symbol()}{inflow:.0f}", 
                    ha="center", 
                    va="bottom", 
                    fontsize=8
                )
                
        # cumulative line
        ax.plot(
            cum["Start Date"], 
            cum["Cumulative Cash Flow"].astype(float), 
            linewidth=2, 
            marker="o", 
            label="Cumulative Cash Flow", 
            color="#2171b5"
        )
        
        _date_formatter(ax)
        ax.set_xlabel(input.time_period())
        ax.set_ylabel(f"Cash Flow ({currency_symbol()})")
        ax.set_title(f"Cumulative Cash Flow by {input.time_period()}")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.legend()
        plt.tight_layout()
        return fig

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
        return csv


app = App(app_ui, server)