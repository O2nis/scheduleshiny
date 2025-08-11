# app.py — Shiny (Python) Project Scheduler - Modern Green Theme (ui.output_data_frame)

from __future__ import annotations

import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from shiny import App, reactive, render, ui

# ------------------------------
# Utilities
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
    "Electrical": {"keywords": ["Electrical", "PV"], "color": "#ff7f0e"},
    "HSE": {"keywords": ["HSE", "Safety"], "color": "#2ca02c"},
    "Instrument": {"keywords": ["Instrument", "SCADA", "I&C", "Automation"], "color": "#d62728"},
    "Management": {"keywords": ["Project Manager", "DCC", "Project Engineer", "Manager"], "color": "#9467bd"},
    "Mechanical": {"keywords": ["Mechanical"], "color": "#8c564b"},
    "Site Supervision": {"keywords": ["Supervisor", "Commissioning"], "color": "#e377c2"},
    "Other": {"keywords": [], "color": "#7f7f7f"},
}

# ------------------------------
# UI
# ------------------------------
app_ui = ui.TagList(
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
        ),
        ui.tags.style(
            """
            :root{
                --primary-green:#059669; --primary-green-light:#10b981; --primary-green-lighter:#34d399; --primary-green-dark:#047857;
                --secondary-green:#d1fae5; --accent-green:#6ee7b7; --mint-green:#f0fdfa; --sage-green:#ecfdf5; --forest-green:#064e3b;
                --background:#f8fafc; --card-bg:#fff; --text-primary:#1f2937; --text-secondary:#6b7280; --border-light:#e5e7eb;
                --shadow:0 4px 6px -1px rgba(0,0,0,.1),0 2px 4px -1px rgba(0,0,0,.06);
                --radius:10px; --radius-lg:12px; --radius-xl:16px;
            }
            body{font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,Arial;background:linear-gradient(135deg,#f0fdfa,#f8fafc);color:var(--text-primary)}
            .sidebar{width:420px;background:var(--card-bg);border:1px solid var(--border-light);border-radius:var(--radius-xl);box-shadow:var(--shadow);padding:26px;margin:16px}
            .btn-primary{background:linear-gradient(135deg,var(--primary-green),var(--primary-green-light))!important;color:#fff!important;border:none!important}
            .card{background:var(--card-bg);border:1px solid var(--border-light);border-radius:var(--radius-xl);box-shadow:var(--shadow)}
            .card-header{background:linear-gradient(135deg,var(--sage-green),var(--mint-green));color:var(--forest-green);font-weight:600}
            /* Data-frame "card" look */
            .df-wrap{border:1px solid var(--border-light);border-radius:14px;box-shadow:var(--shadow);overflow:hidden;margin-bottom:18px}
            .df-title{padding:12px 16px;font-weight:700;background:#f8fafc}
        """
        ),
    ),
    ui.page_sidebar(
        ui.sidebar(
            ui.h4("📊 Data Input Method"),
            ui.input_radio_buttons(
                "input_method", "Choose Input Method", ["Create Table", "Upload File"], selected="Create Table"
            ),
            ui.input_text("currency", "💰 Currency Symbol", value="$"),
            ui.panel_conditional(
                "input.input_method === 'Upload File'",
                ui.input_file("uploaded", "📁 Upload CSV/Excel", multiple=False, accept=[".csv", ".xlsx"]),
                ui.download_button("download_template", "📥 Download CSV Template", class_="btn-info"),
                ui.help_text("Template columns: Service, Role, Hourly Cost, 1..36"),
            ),
            ui.panel_conditional(
                "input.input_method === 'Create Table'",
                ui.h5("🎯 Tasks"),
                ui.layout_columns(
                    ui.input_text("task_name", "Task Name", width="100%"),
                    ui.input_action_button("add_task", "➕ Add", class_="btn-primary"),
                    col_widths=[8, 4],
                ),
                ui.output_ui("current_tasks_list"),
                ui.hr(),
                ui.h5("👥 Roles"),
                ui.layout_columns(
                    ui.input_text("role_name", "Role Name", width="100%"),
                    ui.input_numeric("hourly_cost", "Hourly Cost", value=0, min=0, step=1, width="100%"),
                    col_widths=[6, 6],
                ),
                ui.input_selectize("role_tasks", "Associated Tasks", choices=[], multiple=True),
                ui.input_action_button("add_role", "➕ Add Role", class_="btn-primary"),
                ui.output_ui("current_roles_list"),
                ui.hr(),
                ui.input_numeric("num_periods", "📅 Number of Periods", value=12, min=1, max=36, step=1),
                ui.input_action_button("regen_util", "🎲 Randomize Utilization", class_="btn-secondary"),
                ui.input_action_button("save_schedule", "💾 Save Schedule", class_="btn-primary"),
            ),
            ui.hr(),
            ui.h4("⚙️ Configuration"),
            ui.input_radio_buttons("time_period", "Select Time Period", ["Weeks", "Months"], selected="Months"),
            ui.input_date("project_start", "🗓️ Project Start Date", value=datetime(2025, 1, 1)),
            ui.layout_columns(
                ui.input_numeric("hours_per_day", "⏰ Hours/day", value=9, min=1, max=24),
                ui.input_numeric("work_days_week", "📅 Working days/week", value=5, min=1, max=7),
                ui.input_numeric("work_days_month", "📅 Working days/month", value=22, min=1, max=31),
                col_widths=[4, 4, 4],
            ),
            ui.hr(),
            ui.h4("💳 Cash Flow Settings"),
            ui.layout_columns(
                ui.input_numeric("advance_pct", "Advance Payment (%)", value=0, min=0, max=100, step=1),
                ui.input_numeric("trigger_pct", "Trigger % (balance threshold)", value=-50, min=-100, max=100, step=1),
                col_widths=[6, 6],
            ),
            ui.hr(),
            ui.h4("🏗️ Discipline Management"),
            ui.layout_columns(
                ui.input_text("new_disc", "New Discipline"),
                ui.input_text("new_keys", "Keywords (comma-separated)"),
                col_widths=[6, 6],
            ),
            ui.input_action_button("add_disc", "➕ Add Discipline", class_="btn-primary"),
            ui.layout_columns(
                ui.input_select("mod_disc", "Select Discipline", choices=list(DEFAULT_DISCIPLINES.keys())),
                ui.input_text("new_color", "Hex Color", value="#10b981"),
                ui.input_action_button("upd_color", "🎨 Update Color", class_="btn-secondary"),
                col_widths=[4, 4, 4],
            ),
            ui.output_ui("disc_color_preview"),
            ui.output_data_frame("disc_keywords_table"),
            width=420,
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("🚀 Project Schedule & Cost Visualization Dashboard"),
                ui.markdown("Set inputs on the left, then use the tabs for KPIs, charts and tables."),
                full_screen=True,
            ),
        ),
        ui.navset_pill(
            ui.nav_panel(
                "📝 Data Entry",
                ui.h5("📋 Current Tasks"),
                ui.output_ui("tasks_text"),
                ui.h5("👤 Current Roles"),
                ui.output_data_frame("roles_table"),
                ui.panel_conditional(
                    "input.input_method === 'Create Table'",
                    ui.hr(),
                    ui.h5("⚡ Enter Utilization (0..1) per Role-Task by Period"),
                    ui.output_ui("utilization_ui"),
                ),
                ui.panel_conditional(
                    "input.input_method === 'Upload File'",
                    ui.hr(),
                    ui.h5("📄 File Preview"),
                    ui.output_data_frame("uploaded_head"),
                ),
                ui.hr(),
                ui.h5("👀 Schedule Preview"),
                ui.output_data_frame("schedule_preview"),
            ),
            ui.nav_panel(
                "📊 KPIs",
                ui.output_text_verbatim("kpi_text"),
                ui.h6("⏱️ Total Man-Hours by Period"),
                ui.output_data_frame("manhours_by_period"),
                ui.h6("🏗️ Man-Hours by Discipline"),
                ui.output_data_frame("manhours_by_disc"),
                ui.h6("👥 Man-Hours and Costs by Role"),
                ui.output_data_frame("role_summary"),
            ),
            ui.nav_panel(
                "📈 Charts",
                ui.h5("🔥 Man-Hours by Service (Heatmap)"),
                ui.output_plot("heat_service", height="800px"),
                ui.hr(),
                ui.h5("🌡️ Man-Hours by Role (Heatmap)"),
                ui.output_plot("heat_role", height="800px"),
                ui.hr(),
                ui.h5("📊 Man-Hours per Discipline Over Time (Stacked)"),
                ui.output_plot("hist_disc", height="600px"),
                ui.hr(),
                ui.h5("🍩 Total Man-Hours per Discipline (Donut)"),
                ui.output_plot("donut_disc", height="600px"),
                ui.hr(),
                ui.h5("💰 Total Cost by Period"),
                ui.output_plot("cost_by_period_plot", height="500px"),
                ui.hr(),
                ui.h5("💹 Cumulative Cash Flow by Period"),
                ui.output_plot("cum_cash_plot", height="500px"),
            ),
            ui.nav_panel(
                "📋 Tables",
                ui.div({"class": "df-wrap"},
                    ui.div("💼 Total Cost and Man-Hours by Task", {"class": "df-title"}),
                    ui.output_data_frame("task_summary"),
                ),
                ui.div({"class": "df-wrap"},
                    ui.div("💵 Cost per Period", {"class": "df-title"}),
                    ui.output_data_frame("cost_by_period_table"),
                ),
                ui.div({"class": "df-wrap"},
                    ui.div("📈 Cumulative Cash Flow", {"class": "df-title"}),
                    ui.output_data_frame("cum_cash_table"),
                ),
                ui.hr(),
                ui.div({"class": "df-wrap"},
                    ui.div("🧾 Suggested Invoice Schedule", {"class": "df-title"}),
                    ui.output_data_frame("invoice_table"),
                ),
                ui.download_button("dl_invoice_csv", "📥 Download Invoice CSV", class_="btn-info"),
            ),
        ),
    ),
)

# ------------------------------
# Server
# ------------------------------
def server(input, output, session):
    plt.style.use("seaborn-v0_8-whitegrid")

    # Reactive state
    tasks = reactive.Value([])     # list[str]
    roles = reactive.Value([])     # list[dict{Role, Hourly Cost, Tasks}]
    schedule_df = reactive.Value(pd.DataFrame())
    disciplines = reactive.Value({k: dict(v) for k, v in DEFAULT_DISCIPLINES.items()})

    # Derived
    @reactive.Calc
    def currency_symbol() -> str:
        c = (input.currency() or "$").strip()
        return c if c else "$"

    @reactive.Calc
    def hours_per_period() -> float:
        if input.time_period() == "Months":
            return input.hours_per_day() * input.work_days_month()
        return input.hours_per_day() * input.work_days_week()

    # ---- Small helpers for df outputs (keeps code concise)
    def grid(df: pd.DataFrame):
        # ui.output_data_frame + @render.data_frame will render a DataGrid by default
        return render.DataGrid(df)

    # ------------------ Data Input helpers ------------------
    @render.ui
    def tasks_text():
        t = tasks()
        if not t:
            return ui.div("🔄 No tasks added yet.", class_="text-muted")
        return ui.div(ui.ul(*[ui.li(f"✅ {task}") for task in t]), style="max-height:200px; overflow-y:auto;")

    @render.ui
    def current_tasks_list():
        t = tasks()
        if not t:
            return ui.div()
        return ui.div(
            ui.markdown("**📋 Current Tasks:**"),
            ui.ul(*[ui.li(f"✅ {task}") for task in t]),
            style="margin-bottom:15px; max-height:150px; overflow-y:auto; background:var(--mint-green); padding:12px; border-radius:var(--radius);"
        )

    @render.ui
    def current_roles_list():
        r = roles()
        if not r:
            return ui.div()
        return ui.div(
            ui.markdown("**👥 Current Roles:**"),
            ui.ul(*[
                ui.li(f"👤 **{role['Role']}** ({currency_symbol()}{role['Hourly Cost']}/h) – Tasks: {', '.join(role['Tasks'])}")
                for role in r
            ]),
            style="margin-bottom:15px; max-height:150px; overflow-y:auto; background:var(--sage-green); padding:12px; border-radius:var(--radius);"
        )

    @render.data_frame
    def roles_table():
        r = roles()
        df = (pd.DataFrame(columns=["Role", "Hourly Cost", "Tasks"])
              if not r else
              pd.DataFrame([{"Role": x["Role"], "Hourly Cost": x["Hourly Cost"], "Tasks": ", ".join(x["Tasks"])} for x in r]))
        return grid(df)

    # Add task
    @reactive.Effect
    @reactive.event(input.add_task)
    def _add_task():
        name = (input.task_name() or "").strip()
        if not name:
            ui.notification_show("⚠️ Please enter a task name", duration=3, type="warning"); return
        t = tasks()
        if name not in t:
            t.append(name)
            tasks.set(t)
            ui.update_selectize("role_tasks", choices=tasks(), selected=list(input.role_tasks() or []))
            ui.update_text("task_name", value="")
            ui.notification_show(f"✅ Task '{name}' added", duration=3, type="message")

    # Add / update role
    @reactive.Effect
    @reactive.event(input.add_role)
    def _add_role():
        role_name = (input.role_name() or "").strip()
        if not role_name:
            ui.notification_show("⚠️ Please enter a role name", duration=3, type="warning"); return
        try:
            cost = float(input.hourly_cost() or 0)
        except Exception:
            cost = 0
        if cost <= 0:
            ui.notification_show("⚠️ Hourly cost must be positive", duration=3, type="warning"); return
        assoc = list(input.role_tasks() or [])
        if not assoc:
            ui.notification_show("⚠️ Please select at least one task", duration=3, type="warning"); return

        r = roles()
        existing = next((x for x in r if x["Role"] == role_name), None)
        if existing:
            existing["Hourly Cost"] = cost
            existing["Tasks"] = sorted(set(existing["Tasks"] + assoc))
            ui.notification_show(f"🔄 Role '{role_name}' updated", duration=3, type="message")
        else:
            r.append({"Role": role_name, "Hourly Cost": cost, "Tasks": assoc})
            ui.notification_show(f"✅ Role '{role_name}' added", duration=3, type="message")
        roles.set(r)
        ui.update_text("role_name", value=""); ui.update_numeric("hourly_cost", value=0); ui.update_selectize("role_tasks", selected=[])

    # Discipline management
    @reactive.Effect
    @reactive.event(input.add_disc)
    def _add_disc():
        nd = (input.new_disc() or "").strip()
        nk = (input.new_keys() or "").strip()
        if not nd or not nk:
            ui.notification_show("⚠️ Enter discipline and keywords", duration=3, type="warning"); return
        d = disciplines()
        if nd in d:
            ui.notification_show(f"⚠️ Discipline '{nd}' already exists", duration=3, type="warning"); return
        d[nd] = {"keywords": [k.strip() for k in nk.split(",") if k.strip()], "color": "#%06x" % random.randint(0, 0xFFFFFF)}
        disciplines.set(d)
        ui.update_select("mod_disc", choices=list(d.keys()))
        ui.notification_show(f"✅ Discipline '{nd}' added", duration=3, type="message")

    @reactive.Effect
    @reactive.event(input.upd_color)
    def _upd_color():
        sel = input.mod_disc()
        hexcol = (input.new_color() or "").strip()
        if not sel:
            ui.notification_show("⚠️ Select a discipline", duration=3, type="warning"); return
        if not hexcol.startswith("#") or len(hexcol) != 7:
            ui.notification_show("⚠️ Use hex format like #10b981", duration=3, type="warning"); return
        d = disciplines()
        if sel in d:
            d[sel]["color"] = hexcol
            disciplines.set(d)
            ui.notification_show(f"🎨 Color for '{sel}' set to {hexcol}", duration=3, type="message")

    # Color preview (hex + swatch)
    @render.ui
    def disc_color_preview():
        sel = input.mod_disc()
        d = disciplines()
        col = d.get(sel, {}).get("color", "#10b981")
        return ui.div(
            ui.markdown(f"**Selected:** `{sel or ''}` — **Hex:** `{col}`"),
            ui.div(style=f"width:100%;height:20px;border-radius:6px;margin-top:6px;background:{col};border:1px solid #e5e7eb;"),
            style="padding:10px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;"
        )

    # Discipline ↔ keywords recap
    @render.data_frame
    def disc_keywords_table():
        d = disciplines()
        df = pd.DataFrame([{"Discipline": k, "Keywords": ", ".join(v.get("keywords", [])), "Color": v.get("color", "")} for k, v in d.items()])
        return grid(df)

    # Template download
    @render.download(filename="schedule_template.csv")
    def download_template():
        return create_template_csv()

    # Uploaded file head preview
    @render.data_frame
    def uploaded_head():
        f = input.uploaded()
        if not f:
            return grid(pd.DataFrame(columns=["📁 No file uploaded"]))
        file = f[0]
        try:
            if file["name"].endswith(".xlsx"):
                df = pd.read_excel(file["datapath"], na_values=["", " ", "NA", "NaN"])
            else:
                df = pd.read_csv(file["datapath"], na_values=["", " ", "NA", "NaN"])
            return grid(df.head(50))
        except Exception as e:
            return grid(pd.DataFrame({"❌ Error": [f"Could not read file: {str(e)}"]}))

    # Schedule preview
    @render.data_frame
    def schedule_preview():
        df = schedule_df()
        if df.empty:
            return grid(pd.DataFrame(columns=["📊 No schedule data available"]))
        return grid(df.head(100))

    # ------------------ Utilization dynamic UI ------------------
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
                ui.markdown("**⚠️ No role-task pairs defined.** Add roles with associated tasks first."),
                class_="text-muted",
                style="padding: 16px; border-radius: 10px; border:1px solid var(--border-light); background:#fff;"
            )

        nper = int(input.num_periods())
        if nper <= 0:
            return ui.div("❌ Number of periods must be positive", class_="text-danger")

        rows = []
        header = [
            ui.tags.th("🎯 Service", style="background: var(--primary-green); color: white;"),
            ui.tags.th("👤 Role", style="background: var(--primary-green); color: white;"),
            ui.tags.th("💰 Cost/Hr", style="background: var(--primary-green); color: white;"),
        ] + [ui.tags.th(f"📅 {i}", style="min-width: 56px; background: var(--primary-green-light); color: white;") for i in range(1, nper + 1)]
        rows.append(ui.tags.tr(*header))

        for idx, (svc, role_name, cost) in enumerate(pairs):
            cells = [
                ui.tags.td(svc, style="font-weight:600;"),
                ui.tags.td(role_name, style="font-weight:500;"),
                ui.tags.td(f"{currency_symbol()}{cost:.2f}", style="font-weight:600; color: var(--primary-green);"),
            ]
            for p in range(1, nper + 1):
                input_id = f"util_{idx}_{p}"
                cells.append(
                    ui.tags.td(
                        ui.input_numeric(input_id, None, value=0.0, min=0.0, max=1.0, step=0.1, width="100%"),
                        style="padding:2px;"
                    )
                )
            rows.append(ui.tags.tr(*cells))

        return ui.tags.div(
            {"style": "overflow-x:auto; max-height:600px; overflow-y:auto; border-radius:12px; box-shadow:var(--shadow);"},
            ui.tags.table({"class": "table table-sm table-striped table-bordered"}, *rows),
        )

    @reactive.Effect
    @reactive.event(input.regen_util)
    def _regen_util_values():
        pairs = _role_task_pairs()
        if not pairs:
            ui.notification_show("⚠️ No role-task pairs defined", duration=3, type="warning"); return
        nper = int(input.num_periods())
        if nper <= 0:
            ui.notification_show("⚠️ Number of periods must be positive", duration=3, type="warning"); return
        for idx in range(len(pairs)):
            for p in range(1, nper + 1):
                ui.update_numeric(f"util_{idx}_{p}", value=round(random.uniform(0, 0.9), 1))
        ui.notification_show("🎲 Utilization values regenerated!", duration=3, type="message")

    # Save schedule (manual creation) — robust to untouched inputs
    @reactive.Effect
    @reactive.event(input.save_schedule)
    def _save_schedule():
        if input.input_method() != "Create Table":
            return
        pairs = _role_task_pairs()
        if not pairs:
            ui.notification_show("⚠️ No role-task pairs defined", duration=3, type="warning"); return
        nper = int(input.num_periods())
        if nper <= 0:
            ui.notification_show("⚠️ Number of periods must be positive", duration=3, type="warning"); return

        data = []
        for idx, (svc, role_name, cost) in enumerate(pairs):
            row = {"Service": svc, "Role": role_name, "Hourly Cost": cost}
            for p in range(1, nper + 1):
                input_id = f"util_{idx}_{p}"
                try:
                    val = input[input_id]()  # may be None
                except KeyError:
                    val = 0.0
                row[str(p)] = float(val or 0.0)
            data.append(row)

        df = pd.DataFrame(data)
        schedule_df.set(df)
        ui.notification_show("💾 Schedule saved!", duration=3, type="message")

    # Load uploaded file
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
            ui.notification_show("📁 File uploaded!", duration=3, type="message")
        except Exception as e:
            ui.notification_show(f"❌ Error reading file: {str(e)}", duration=5, type="error")

    # ------------------ Core computations ------------------
    @reactive.Calc
    def processed():
        df = schedule_df()
        if df is None or df.empty:
            return None

        df = df.copy()
        df.columns = df.columns.str.strip()
        service_col = "Service"; role_col = "Role"; cost_col = "Hourly Cost"
        all_period_columns = [col for col in df.columns if str(col).isdigit()]

        # clean numeric
        for col in all_period_columns:
            df[col] = pd.to_numeric(pd.Series(df[col]).astype(str).str.replace(",", ".").str.replace("%", ""), errors="coerce")
        df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0)

        # period set (tight range)
        non_empty_periods = df[all_period_columns].columns[(df[all_period_columns].fillna(0) != 0).any()]
        if len(non_empty_periods) == 0:
            period_columns = list(all_period_columns)
        else:
            nums = [int(c) for c in non_empty_periods]
            mi, ma = min(nums), max(nums)
            period_columns = [str(i) for i in range(mi, ma + 1) if str(i) in all_period_columns]

        # dates
        period_dates = {c: get_period_start_end_dates(int(c), input.project_start(), input.time_period())[0] for c in period_columns}

        # disciplines
        disc_map = disciplines()
        df["Discipline"] = df[role_col].apply(lambda x: map_discipline(x, disc_map))

        # man-hours & cost
        hpp = hours_per_period()
        total_man_hours = df[period_columns].sum().sum() * hpp
        df["Cost"] = df[period_columns].sum(axis=1) * hpp * df[cost_col]
        total_cost = float(df["Cost"].sum())
        avg_cost = (total_cost / total_man_hours) if total_man_hours > 0 else 0.0

        man_hours_by_period = (df[period_columns].sum() * hpp)
        man_hours_by_discipline = df.groupby("Discipline")[period_columns].sum().sum(axis=1) * hpp
        man_hours_by_role = df.groupby(role_col)[period_columns].sum().sum(axis=1) * hpp
        cost_by_role = df.groupby(role_col).apply(lambda x: (x[period_columns].sum(axis=1) * hpp * x[cost_col]).sum(), include_groups=False)
        man_hours_by_task = df.groupby(service_col)[period_columns].sum().sum(axis=1) * hpp
        cost_by_task = df.groupby(service_col).apply(lambda x: (x[period_columns].sum(axis=1) * hpp * x[cost_col]).sum(), include_groups=False)

        # discipline time series
        disc_list = sorted(df["Discipline"].unique())
        disc_colors = {d: disc_map.get(d, {}).get("color", "#000000") for d in disc_list}
        disc_time = pd.DataFrame(index=disc_list, columns=period_columns)
        for dsc in disc_list:
            disc_rows = df[df["Discipline"] == dsc]
            for col in period_columns:
                vals = disc_rows[col].fillna(0).clip(lower=0, upper=1)
                disc_time.loc[dsc, col] = (vals * hpp).sum()
        disc_time = disc_time.apply(pd.to_numeric, errors='coerce').fillna(0)
        disc_time_t = disc_time.transpose()
        disc_time_t.index = [period_dates[c] for c in disc_time_t.index]
        disc_time_t.sort_index(inplace=True)

        # Cost by period (positive)
        cost_by_period = pd.Series(0.0, index=period_columns)
        for col in period_columns:
            pct = df[col].fillna(0).clip(0, 1)
            mh = pct * hpp
            cost_by_period[col] = (mh.mul(df[cost_col], axis=0)).sum()

        # Period costs negative for cumulative model
        period_costs = -cost_by_period.copy()

        # Cash flow model
        advance_payment = total_cost * (float(input.advance_pct()) / 100.0)
        cash_flows = pd.Series(0.0, index=period_columns)
        if period_columns:
            cash_flows[period_columns[0]] = advance_payment

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
                invoice_amount = min(invoice_amount, total_cost - total_invoiced)
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
            "Start Date": [period_dates[c] for c in period_columns],
            "Cash Flow": cash_flows.values,
            "Cumulative Cash Flow": cumulative.values,
        })

        return dict(
            df=df, service_col=service_col, role_col=role_col, cost_col=cost_col,
            period_columns=period_columns, period_dates=period_dates,
            total_man_hours=float(total_man_hours), total_cost=float(total_cost), avg_cost=float(avg_cost),
            man_hours_by_period=man_hours_by_period, man_hours_by_discipline=man_hours_by_discipline,
            man_hours_by_role=man_hours_by_role, cost_by_role=cost_by_role,
            disc_time_t=disc_time_t, disc_colors=disc_colors, cost_by_period=cost_by_period,
            man_hours_by_task=man_hours_by_task, cost_by_task=cost_by_task,
            cumulative_df=cum_df, period_costs=period_costs,
        )

    # ------------------ KPIs & Tables ------------------
    @render.text
    def kpi_text():
        P = processed()
        if not P: return "📊 Upload or create a schedule to see KPIs."
        return (
            f"🕒 Total Man-Hours: {P['total_man_hours']:,.2f}\n"
            f"💰 Total Cost: {currency_symbol()}{P['total_cost']:,.2f}\n"
            f"📈 Average Hourly Cost: {currency_symbol()}{P['avg_cost']:.2f}"
        )

    @render.data_frame
    def manhours_by_period():
        P = processed()
        if not P: return grid(pd.DataFrame(columns=["No data"]))
        df = pd.DataFrame({
            "Period": P["period_columns"],
            "Start Date": [P["period_dates"][c] for c in P["period_columns"]],
            "Man-Hours": P["man_hours_by_period"].values,
        })
        return grid(df)

    @render.data_frame
    def manhours_by_disc():
        P = processed()
        if not P: return grid(pd.DataFrame(columns=["No data"]))
        df = P["man_hours_by_discipline"].reset_index().rename(columns={0: "Man-Hours", "index": "Discipline"})
        return grid(df)

    @render.data_frame
    def role_summary():
        P = processed()
        if not P: return grid(pd.DataFrame(columns=["No data"]))
        df = pd.DataFrame({
            "Role": P["man_hours_by_role"].index,
            "Man-Hours": P["man_hours_by_role"].values,
            "Cost": P["cost_by_role"].values,
        })
        df["Cost"] = df["Cost"].map(lambda x: f"{currency_symbol()}{x:,.2f}")
        return grid(df)

    @render.data_frame
    def task_summary():
        P = processed()
        if not P: return grid(pd.DataFrame(columns=["No data"]))
        man = P["man_hours_by_task"]; cost = P["cost_by_task"]
        df = pd.DataFrame({"Service": man.index, "Man-Hours": man.values, "Cost": cost.values})
        df["Cost"] = df["Cost"].map(lambda x: f"{currency_symbol()}{x:,.2f}")
        return grid(df)

    @render.data_frame
    def cost_by_period_table():
        P = processed()
        if not P: return grid(pd.DataFrame(columns=["No data"]))
        df = pd.DataFrame({
            "Period": P["period_columns"],
            "Start Date": [P["period_dates"][c] for c in P["period_columns"]],
            "Cost": P["cost_by_period"].values,
        })
        df["Cost"] = df["Cost"].map(lambda x: f"{currency_symbol()}{x:,.2f}")
        return grid(df)

    @render.data_frame
    def cum_cash_table():
        P = processed()
        if not P: return grid(pd.DataFrame(columns=["No data"]))
        df = P["cumulative_df"].copy()
        df["Cash Flow"] = df["Cash Flow"].map(lambda x: f"{currency_symbol()}{x:,.2f}")
        df["Cumulative Cash Flow"] = df["Cumulative Cash Flow"].map(lambda x: f"{currency_symbol()}{x:,.2f}")
        return grid(df)

    @render.data_frame
    def invoice_table():
        P = processed()
        if not P: return grid(pd.DataFrame(columns=["No data"]))
        rows = []; inv_no = 1
        for _, row in P["cumulative_df"].iterrows():
            cf = float(row["Cash Flow"])
            if cf > 0:
                total_cost = P["total_cost"]
                pct = (cf / total_cost * 100.0) if total_cost > 0 else 0.0
                rows.append({
                    "Invoice": f"Invoice {inv_no}",
                    "Submission Date": row["Start Date"],
                    "Amount": f"{currency_symbol()}{cf:,.2f}",
                    "Percentage of Total Cost": f"{pct:.2f}%",
                })
                inv_no += 1
        return grid(pd.DataFrame(rows))

    # ------------------ Plots ------------------
    def _date_formatter(ax):
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45, fontsize=9, ha="right")
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        plt.tight_layout()

    @render.plot
    def heat_service():
        P = processed()
        if not P: return
        df = P["df"]; period_cols = P["period_columns"]; hpp = hours_per_period()

        rows = []
        for task in df[P["service_col"]].unique():
            sub = df[df[P["service_col"]] == task]
            for c in period_cols:
                s = sub[c].sum()
                if pd.notna(s):
                    rows.append({"Task": task, "Period": c, "Man_Hours": s * hpp})
        gdf = pd.DataFrame(rows)
        if gdf.empty: return

        pivot = gdf.pivot(index="Task", columns="Period", values="Man_Hours").fillna(0)
        for c in period_cols:
            if c not in pivot.columns: pivot[c] = 0.0

        order_map = get_earliest_period(df, P["service_col"], period_cols)
        tasks_ordered = sorted(pivot.index, key=lambda x: order_map.get(x, 10**9))
        periods_ordered = sorted([c for c in period_cols], key=int)
        pivot = pivot.loc[tasks_ordered, periods_ordered]

        fig, ax = plt.subplots(figsize=(max(12, len(periods_ordered)), max(8, len(tasks_ordered)*0.7)))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("white_emerald", ["#ffffff", "#10b981"])
        annot = pivot.applymap(lambda v: "" if v == 0 else f"{v:.0f}")

        sns.heatmap(pivot, cmap=cmap, annot=annot, fmt="", cbar_kws={"label": "Man-Hours"},
                    ax=ax, linewidths=0.5, linecolor="#e5e7eb", annot_kws={"size":8, "color":"#064e3b"},
                    vmin=0, vmax=max(pivot.values.max()*1.2 or 1, 20))

        ax.set_xticklabels([P["period_dates"][p].strftime("%Y-%m-%d") for p in periods_ordered], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(tasks_ordered, rotation=0, fontsize=9)
        ax.set_xlabel(input.time_period()); ax.set_ylabel("Tasks")
        ax.set_title(f"Man-Hours by Service ({input.time_period()})")
        max_task_len = max((len(s) for s in tasks_ordered), default=10)
        left = min(0.72, 0.14 + 0.013 * max_task_len)
        bottom = min(0.42, 0.18 + 0.005 * len(periods_ordered))
        fig.subplots_adjust(left=left, bottom=bottom, right=0.98, top=0.92)
        return fig

    @render.plot
    def heat_role():
        P = processed()
        if not P: return
        df = P["df"]; period_cols = P["period_columns"]; hpp = hours_per_period()

        rows = []
        for role in df[P["role_col"]].unique():
            sub = df[df[P["role_col"]] == role]
            for c in period_cols:
                s = sub[c].sum()
                if pd.notna(s):
                    rows.append({"Role": role, "Period": c, "Man_Hours": s * hpp})
        gdf = pd.DataFrame(rows)
        if gdf.empty: return

        pivot = gdf.pivot(index="Role", columns="Period", values="Man_Hours").fillna(0)
        for c in period_cols:
            if c not in pivot.columns: pivot[c] = 0.0

        order_map = get_earliest_period(df, P["role_col"], period_cols)
        roles_ordered = sorted(pivot.index, key=lambda x: order_map.get(x, 10**9))
        periods_ordered = sorted([c for c in period_cols], key=int)
        pivot = pivot.loc[roles_ordered, periods_ordered]

        fig, ax = plt.subplots(figsize=(max(12, len(periods_ordered)), max(8, len(roles_ordered)*0.7)))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("white_teal", ["#ffffff", "#34d399"])
        annot = pivot.applymap(lambda v: "" if v == 0 else f"{v:.0f}")

        sns.heatmap(pivot, cmap=cmap, annot=annot, fmt="", cbar_kws={"label": "Man-Hours"},
                    ax=ax, linewidths=0.5, linecolor="#e5e7eb", annot_kws={"size":8, "color":"#064e3b"},
                    vmin=0, vmax=max(pivot.values.max()*1.2 or 1, 20))

        ax.set_xticklabels([P["period_dates"][p].strftime("%Y-%m-%d") for p in periods_ordered], rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(roles_ordered, rotation=0, fontsize=9)
        ax.set_xlabel(input.time_period()); ax.set_ylabel("Roles")
        ax.set_title(f"Man-Hours by Role ({input.time_period()})")
        max_role_len = max((len(r) for r in roles_ordered), default=10)
        left = min(0.74, 0.14 + 0.013 * max_role_len)
        bottom = min(0.42, 0.18 + 0.005 * len(periods_ordered))
        fig.subplots_adjust(left=left, bottom=bottom, right=0.98, top=0.92)
        return fig

    @render.plot
    def hist_disc():
        P = processed()
        if not P: return
        dt = P["disc_time_t"]
        if dt.empty: return

        disc_cols = list(dt.columns)
        palette = [P["disc_colors"][d] for d in disc_cols]

        fig, ax = plt.subplots(figsize=(14, 10))
        bottom = np.zeros(len(dt))
        bar_width = 15 if input.time_period() == "Months" else 2

        for i, d in enumerate(disc_cols):
            vals = dt[d].values.astype(float)
            bars = ax.bar(dt.index, vals, bottom=bottom, label=d, color=palette[i], width=bar_width, alpha=0.95)
            threshold = 10 if input.time_period() == "Months" else 2
            for b in bars:
                h = b.get_height()
                if h > threshold:
                    ax.text(b.get_x()+b.get_width()/2, b.get_y()+h/2, f"{h:.0f}",
                            ha="center", va="center", fontsize=8, rotation=90,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            bottom += vals

        _date_formatter(ax)
        ax.set_xlabel(input.time_period()); ax.set_ylabel("Man-Hours")
        ax.set_title(f"Man-Hours per Discipline Over {input.time_period()}")
        ax.legend(title="Discipline", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        plt.tight_layout(pad=2.0)
        return fig

    @render.plot
    def donut_disc():
        P = processed()
        if not P: return
        dt = P["disc_time_t"]
        if dt.empty: return

        totals = dt.sum(axis=0)
        labels = totals.index
        sizes = (totals / totals.sum() * 100.0).values if totals.sum() != 0 else np.zeros(len(totals))
        colors = [P["disc_colors"][d] for d in labels]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pie(
            sizes, labels=labels, colors=colors, startangle=90,
            autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
            pctdistance=0.85, labeldistance=1.05,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"color": "black", "fontsize": 12}, radius=1.2
        )
        center = plt.Circle((0, 0), 0.70, fc="white"); ax.add_artist(center)
        ax.axis("equal"); ax.set_title("Total Man-Hours per Discipline")
        plt.tight_layout(); return fig

    @render.plot
    def cost_by_period_plot():
        P = processed()
        if not P: return
        df = pd.DataFrame({
            "Start Date": [P["period_dates"][c] for c in P["period_columns"]],
            "Cost": P["cost_by_period"].values,
        })
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 15 if input.time_period() == "Months" else 2
        bars = ax.bar(df["Start Date"], df["Cost"].astype(float), width=bar_width, color="#10b981")
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h, f"{currency_symbol()}{h:,.0f}", ha="center", va="bottom", fontsize=8)
        _date_formatter(ax)
        ax.set_xlabel(input.time_period()); ax.set_ylabel(f"Cost ({currency_symbol()})")
        ax.set_title(f"Total Cost by {input.time_period()}"); plt.tight_layout(); return fig

    @render.plot
    def cum_cash_plot():
        P = processed()
        if not P: return
        cum = P["cumulative_df"].copy(); period_costs = P["period_costs"]

        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 15 if input.time_period() == "Months" else 2
        ax.bar(cum["Start Date"], period_costs.astype(float), width=bar_width, color="#f59e0b", label="Costs")
        ax.bar(cum["Start Date"], cum["Cash Flow"].astype(float), width=bar_width, color="#10b981", label="Inflows")
        ax.plot(cum["Start Date"], cum["Cumulative Cash Flow"].astype(float), linewidth=2.2, marker="o", color="#064e3b", label="Cumulative")

        _date_formatter(ax)
        ax.set_xlabel(input.time_period()); ax.set_ylabel(f"Cash Flow ({currency_symbol()})")
        ax.set_title(f"Cumulative Cash Flow by {input.time_period()}")
        ax.axhline(0, color="black", linestyle="--", linewidth=1); ax.legend(); plt.tight_layout(); return fig

    # Invoice CSV download
    @render.download(filename="invoice_schedule.csv")
    def dl_invoice_csv():
        P = processed()
        if not P: return b""
        rows = []; inv_no = 1
        for _, row in P["cumulative_df"].iterrows():
            cf = float(row["Cash Flow"])
            if cf > 0:
                pct = (cf / P["total_cost"] * 100.0) if P["total_cost"] > 0 else 0.0
                rows.append({
                    "Invoice": f"Invoice {inv_no}",
                    "Submission Date": row["Start Date"].strftime("%Y-%m-%d"),
                    "Amount": f"{currency_symbol()}{cf:,.2f}",
                    "Percentage of Total Cost": f"{pct:.2f}%",
                }); inv_no += 1
        return pd.DataFrame(rows).to_csv(index=False)

app = App(app_ui, server)
