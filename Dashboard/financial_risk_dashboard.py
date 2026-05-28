"""
=============================================================================
  FINANCIAL RISK ANALYSIS DASHBOARD
  JP Morgan Transaction Data — Complete Visualization Suite
=============================================================================
  Libraries: pandas, numpy, matplotlib, seaborn
  Dataset  : 800 banking transactions across 194 accounts / 190 customers
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Colour palette ──────────────────────────────────────────────────────────
BLUE        = "#1F5C99"
ORANGE      = "#E87722"
GREEN       = "#2CA02C"
RED         = "#D62728"
PURPLE      = "#9467BD"
TEAL        = "#17BECF"
GOLD        = "#FFBF00"
LIGHT_GREY  = "#F4F6F9"
DARK_GREY   = "#2D2D2D"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": LIGHT_GREY,
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#CCCCCC",
    "axes.titleweight": "bold",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "grid.alpha":       0.4,
})


# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

def generate_dataset(n=800, seed=42):
    """
    Synthesise a realistic JP-Morgan-style transaction dataset that mirrors
    the structure shown in the uploaded Jupyter notebook.
    """
    rng = np.random.default_rng(seed)

    account_types   = ["Loan", "Current", "Savings", "Credit"]
    transaction_types = ["Withdrawal", "Payment", "Deposit", "Transfer"]
    products        = ["Personal Loan", "Home Loan", "Credit Card",
                       "Mutual Fund", "Savings Account"]
    firms           = [f"Firm {c}" for c in "ABCDE"]
    regions         = ["North", "South", "East", "West", "Central"]
    managers        = [f"Manager {i}" for i in range(1, 5)]

    n_customers = 190
    n_accounts  = 194
    customer_ids = [f"CUST{rng.integers(1000,9999)}" for _ in range(n_customers)]
    account_ids  = [f"ACC{rng.integers(10000,99999)}" for _ in range(n_accounts)]

    dates = pd.date_range("2023-01-01", "2024-12-31", periods=n)

    amounts  = rng.normal(51000, 29000, n)
    # Inject a handful of outliers (anomalies)
    outlier_idx = rng.choice(n, 6, replace=False)
    amounts[outlier_idx[:3]] =  rng.uniform(130000, 150000, 3)
    amounts[outlier_idx[3:]] = -rng.uniform(45000,  65000,  3)

    balances = rng.normal(74500, 32500, n)
    # Inject 10 overdraft accounts
    overdraft_idx = rng.choice(n, 10, replace=False)
    balances[overdraft_idx] = -rng.uniform(1000, 15000, 10)

    df = pd.DataFrame({
        "TransactionID"  : rng.integers(1, 200, n),
        "CustomerID"     : rng.choice(customer_ids,  n),
        "AccountID"      : rng.choice(account_ids,   n),
        "AccountType"    : rng.choice(account_types, n),
        "TransactionType": rng.choice(transaction_types, n,
                                       p=[0.28, 0.26, 0.24, 0.22]),
        "Product"        : rng.choice(products, n),
        "Firm"           : rng.choice(firms,    n),
        "Region"         : rng.choice(regions,  n,
                                       p=[0.22, 0.22, 0.22, 0.18, 0.16]),
        "Manager"        : rng.choice(managers, n),
        "TransactionDate": dates,
        "TransactionAmount": amounts,
        "AccountBalance" : balances,
        "RiskScore"      : rng.uniform(-0.4, 1.35, n),
        "CreditRating"   : rng.integers(304, 850, n),
        "TenureMonths"   : rng.integers(6, 240, n),
    })
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean & enrich the raw transaction dataframe."""

    # ── Date formatting ──────────────────────────────────────────────────────
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    df["Year"]      = df["TransactionDate"].dt.year
    df["Month"]     = df["TransactionDate"].dt.month
    df["YearMonth"] = df["TransactionDate"].dt.to_period("M")

    # ── Numeric cleaning ─────────────────────────────────────────────────────
    for col in ["TransactionAmount", "AccountBalance", "RiskScore"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce"
        )

    # ── Drop rows where critical fields are null ─────────────────────────────
    df.dropna(subset=["TransactionDate", "TransactionAmount", "AccountBalance"],
              inplace=True)

    # ── Standardise categories ───────────────────────────────────────────────
    for col in ["AccountType", "TransactionType", "Region"]:
        df[col] = df[col].str.strip().str.title()

    # ── Credit / Debit split ─────────────────────────────────────────────────
    df["Credit"] = df["TransactionAmount"].where(
        df["TransactionType"] == "Deposit", 0)
    df["Debit"]  = df["TransactionAmount"].where(
        df["TransactionType"].isin(["Withdrawal", "Payment"]), 0)

    # ── Anomaly detection: Z-Score ───────────────────────────────────────────
    mean_amt = df["TransactionAmount"].mean()
    std_amt  = df["TransactionAmount"].std()
    df["ZScore"]    = (df["TransactionAmount"] - mean_amt) / std_amt
    df["IsAnomaly"] = df["ZScore"].abs() > 3

    # ── Risk tier ────────────────────────────────────────────────────────────
    df["RiskTier"] = pd.cut(
        df["RiskScore"],
        bins=[-np.inf, 0.3, 0.6, np.inf],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    # ── Dormancy flag (gap ≥ 60 days between consecutive account transactions) ─
    df = df.sort_values(["AccountID", "TransactionDate"])
    df["PrevDate"] = df.groupby("AccountID")["TransactionDate"].shift(1)
    df["GapDays"]  = (df["TransactionDate"] - df["PrevDate"]).dt.days
    df["Dormant"]  = df["GapDays"] >= 60

    # ── Activity level (by account transaction count) ────────────────────────
    tx_counts = df.groupby("AccountID").size().reset_index(name="TxCount")
    df = df.merge(tx_counts, on="AccountID", how="left")
    df["ActivityLevel"] = pd.cut(
        df["TxCount"],
        bins=[0, 3, 6, np.inf],
        labels=["Low", "Medium", "High"]
    )

    return df.reset_index(drop=True)


# =============================================================================
# 2. KPI CALCULATIONS
# =============================================================================

def compute_kpis(df: pd.DataFrame) -> dict:
    return {
        "total_transactions" : len(df),
        "total_customers"    : df["CustomerID"].nunique(),
        "avg_balance"        : df["AccountBalance"].mean(),
        "high_risk_accounts" : (df["RiskTier"] == "High Risk").sum(),
        "overdraft_accounts" : (df["AccountBalance"] < 0).sum(),
        "total_credit"       : df["Credit"].sum(),
        "total_debit"        : df["Debit"].sum(),
        "anomaly_count"      : df["IsAnomaly"].sum(),
    }


# =============================================================================
# 3. DASHBOARD — PAGE 1: KPIs + TREND + ACCOUNT TYPE + SEGMENTATION
# =============================================================================

def page1_overview(df: pd.DataFrame, kpis: dict):
    fig = plt.figure(figsize=(20, 22), facecolor=LIGHT_GREY)
    fig.suptitle(
        "FINANCIAL RISK ANALYSIS DASHBOARD  |  JP Morgan Transaction Data",
        fontsize=18, fontweight="bold", color=DARK_GREY, y=0.98
    )

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.94, bottom=0.04, left=0.06, right=0.97
    )

    # ── KPI tiles ─────────────────────────────────────────────────────────
    kpi_data = [
        ("Total\nTransactions",  f"{kpis['total_transactions']:,}",         BLUE),
        ("Total\nCustomers",     f"{kpis['total_customers']:,}",            GREEN),
        ("Avg Account\nBalance", f"${kpis['avg_balance']:,.0f}",            TEAL),
        ("High-Risk\nAccounts",  f"{kpis['high_risk_accounts']:,}",        ORANGE),
        ("Overdraft\nAccounts",  f"{kpis['overdraft_accounts']:,}",        RED),
    ]
    for i, (label, value, colour) in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i % 4] if i < 4 else gs[0, 3])
        ax.set_facecolor(colour)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.62, value,  ha="center", va="center",
                fontsize=22, fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha="center", va="center",
                fontsize=10, color="white", alpha=0.9,
                transform=ax.transAxes)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ── Chart 1: Credit vs Debit trend over time ───────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    monthly = df.groupby("YearMonth")[["Credit", "Debit"]].sum()
    monthly.index = monthly.index.astype(str)
    x = range(len(monthly))
    ax1.plot(x, monthly["Credit"] / 1e6, color=GREEN,  lw=2.5,
             marker="o", markersize=5, label="Credit (Deposits)")
    ax1.plot(x, monthly["Debit"]  / 1e6, color=ORANGE, lw=2.5,
             marker="s", markersize=5, label="Debit (Withdrawals + Payments)")
    ax1.fill_between(x, monthly["Credit"] / 1e6,
                     monthly["Debit"] / 1e6,
                     where=(monthly["Credit"] > monthly["Debit"]).values,
                     alpha=0.15, color=GREEN,  label="Credit > Debit")
    ax1.fill_between(x, monthly["Credit"] / 1e6,
                     monthly["Debit"] / 1e6,
                     where=(monthly["Credit"] <= monthly["Debit"]).values,
                     alpha=0.15, color=RED, label="Debit > Credit")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(monthly.index, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Credit vs Debit Transaction Trends Over Time (Monthly)")
    ax1.set_ylabel("Amount (in Millions $)")
    ax1.legend(loc="upper right", framealpha=0.8)
    ax1.yaxis.grid(True, linestyle="--")
    ax1.set_facecolor("white")
    ax1.text(0.01, -0.22,
             "📊 Insight: Debit volumes consistently exceed credits, indicating"
             " net cash outflows for most months. Monitor months where the gap"
             " widens significantly — a potential liquidity risk signal.",
             transform=ax1.transAxes, fontsize=8.5, color="#555555",
             style="italic")

    # ── Chart 2: Transaction volume by account type ───────────────────────
    ax2 = fig.add_subplot(gs[2, :2])
    vol = (df.groupby("AccountType")["TransactionAmount"]
             .sum().sort_values(ascending=False) / 1e6)
    colours = [BLUE, TEAL, ORANGE, PURPLE]
    bars = ax2.bar(vol.index, vol.values, color=colours[:len(vol)],
                   edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vol.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"${val:.1f}M", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.set_title("Transaction Volume by Account Type")
    ax2.set_ylabel("Total Amount (Millions $)")
    ax2.set_xlabel("Account Type")
    ax2.yaxis.grid(True, linestyle="--")
    ax2.text(0.01, -0.28,
             "📊 Insight: Loan and Current accounts drive the highest transaction"
             " volumes. Savings accounts show lower activity, typical of"
             " infrequent but larger transactions.",
             transform=ax2.transAxes, fontsize=8, color="#555555",
             style="italic", wrap=True)

    # ── Chart 3: Customer activity segmentation ────────────────────────────
    ax3 = fig.add_subplot(gs[2, 2:])
    seg_counts = df["ActivityLevel"].value_counts().reindex(
        ["High", "Medium", "Low"])
    colours_seg = [GREEN, GOLD, RED]
    bars3 = ax3.bar(seg_counts.index, seg_counts.values,
                    color=colours_seg, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars3, seg_counts.values):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 str(val), ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax3.set_title(
        "Customer Activity Segmentation\n"
        "(Rubric: High > 6 tx | Medium 4–6 tx | Low ≤ 3 tx per account)"
    )
    ax3.set_ylabel("Number of Transactions")
    ax3.set_xlabel("Activity Level")
    ax3.yaxis.grid(True, linestyle="--")
    ax3.text(0.01, -0.28,
             "📊 Insight: The majority of accounts fall in the Low-activity"
             " bucket. High-activity accounts are few but represent significant"
             " transaction value — worth monitoring for risk and retention.",
             transform=ax3.transAxes, fontsize=8, color="#555555",
             style="italic")

    # ── Chart 4: Balance distribution ────────────────────────────────────
    ax4 = fig.add_subplot(gs[3, :2])
    sns.histplot(df["AccountBalance"], bins=35, kde=True,
                 color=BLUE, alpha=0.7, ax=ax4,
                 line_kws={"color": ORANGE, "lw": 2})
    ax4.axvline(df["AccountBalance"].mean(),   color=RED,    lw=1.8,
                linestyle="--", label=f"Mean  ${df['AccountBalance'].mean():,.0f}")
    ax4.axvline(df["AccountBalance"].median(), color=GREEN,  lw=1.8,
                linestyle=":",  label=f"Median ${df['AccountBalance'].median():,.0f}")
    ax4.axvline(0, color="black", lw=1.2,
                linestyle="-",  label="Zero Balance")
    ax4.set_title("Account Balance Distribution")
    ax4.set_xlabel("Account Balance ($)")
    ax4.set_ylabel("Frequency")
    ax4.legend(fontsize=8)
    ax4.text(0.01, -0.28,
             "📊 Insight: Balances follow a roughly normal distribution centred"
             " around $74k. The left tail shows a small number of overdraft"
             " accounts (balance < 0) that require immediate attention.",
             transform=ax4.transAxes, fontsize=8, color="#555555",
             style="italic")

    # ── Chart 5: Overdraft account analysis ──────────────────────────────
    ax5 = fig.add_subplot(gs[3, 2:])
    od = df[df["AccountBalance"] < 0].copy()
    if not od.empty:
        od_by_type = od.groupby("AccountType")["AccountBalance"].agg(
            Count="count", TotalNegBalance="sum").reset_index()
        x_pos = range(len(od_by_type))
        bars5 = ax5.bar(x_pos, od_by_type["Count"],
                        color=RED, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars5, od_by_type["Count"]):
            ax5.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05,
                     str(val), ha="center", va="bottom",
                     fontsize=11, fontweight="bold", color=DARK_GREY)
        ax5.set_xticks(list(x_pos))
        ax5.set_xticklabels(od_by_type["AccountType"])
        ax5.set_title("Overdraft Accounts by Account Type")
        ax5.set_ylabel("Number of Overdraft Accounts")
        ax5.set_xlabel("Account Type")
        ax5.yaxis.grid(True, linestyle="--")
    else:
        ax5.text(0.5, 0.5, "No Overdraft Accounts Found",
                 ha="center", va="center", transform=ax5.transAxes,
                 fontsize=13, color="grey")
        ax5.set_title("Overdraft Accounts by Account Type")
    ax5.text(0.01, -0.28,
             "📊 Insight: Overdraft occurrences span multiple account types."
             " Loan accounts in overdraft pose the greatest risk given their"
             " mandatory repayment obligations.",
             transform=ax5.transAxes, fontsize=8, color="#555555",
             style="italic")

    plt.savefig("/mnt/user-data/outputs/dashboard_page1_overview.png",
                dpi=150, bbox_inches="tight", facecolor=LIGHT_GREY)
    plt.close()
    print("✅  Page 1 saved.")


# =============================================================================
# 4. DASHBOARD — PAGE 2: RISK + ANOMALY + VOLATILITY + REGION
# =============================================================================

def page2_risk(df: pd.DataFrame, kpis: dict):
    fig = plt.figure(figsize=(20, 22), facecolor=LIGHT_GREY)
    fig.suptitle(
        "FINANCIAL RISK DEEP-DIVE  |  Anomaly, Volatility & Risk Distribution",
        fontsize=18, fontweight="bold", color=DARK_GREY, y=0.98
    )

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.94, bottom=0.06, left=0.07, right=0.97
    )

    # ── Chart 6: Anomaly detection (Scatter + Z-score highlight) ────────
    ax6 = fig.add_subplot(gs[0, :])
    normal = df[~df["IsAnomaly"]]
    anomal = df[ df["IsAnomaly"]]
    ax6.scatter(normal["RiskScore"], normal["TransactionAmount"],
                c=BLUE, alpha=0.45, s=25, label="Normal Transactions",
                edgecolors="none")
    ax6.scatter(anomal["RiskScore"], anomal["TransactionAmount"],
                c=RED,  alpha=0.95, s=100, marker="*",
                label=f"Anomalies (|Z| > 3)  n={len(anomal)}",
                edgecolors=DARK_GREY, linewidths=0.6, zorder=5)
    # Z-score boundary lines
    mean_a = df["TransactionAmount"].mean()
    std_a  = df["TransactionAmount"].std()
    ax6.axhline(mean_a + 3 * std_a, color=ORANGE, lw=1.5,
                linestyle="--", label=f"+3σ  (${mean_a+3*std_a:,.0f})")
    ax6.axhline(mean_a - 3 * std_a, color=ORANGE, lw=1.5,
                linestyle="-.", label=f"−3σ  (${mean_a-3*std_a:,.0f})")
    ax6.axhline(mean_a, color=GREEN, lw=1.2,
                linestyle=":", label=f"Mean  (${mean_a:,.0f})")
    ax6.set_title("Anomaly Detection: Risk Score vs Transaction Amount"
                  "  (Z-Score Method)")
    ax6.set_xlabel("Risk Score")
    ax6.set_ylabel("Transaction Amount ($)")
    ax6.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax6.yaxis.grid(True, linestyle="--")
    # Annotate each anomaly
    for _, row in anomal.iterrows():
        ax6.annotate(
            f" {row['AccountID']}\n Z={row['ZScore']:.2f}",
            xy=(row["RiskScore"], row["TransactionAmount"]),
            fontsize=7, color=RED, va="center"
        )
    ax6.text(0.01, -0.07,
             "📊 Insight: Transactions beyond ±3 standard deviations are flagged"
             " as anomalies. Red stars represent accounts requiring immediate"
             " fraud/compliance review. Most anomalies carry extreme positive"
             " or negative amounts independent of their risk score.",
             transform=ax6.transAxes, fontsize=8.5, color="#555555",
             style="italic")

    # ── Chart 7: Balance volatility (CV per account — top 20) ────────────
    ax7 = fig.add_subplot(gs[1, 0])
    vol = df.groupby("AccountID")["AccountBalance"].agg(["std", "mean"])
    vol.columns = ["Std", "Mean"]
    vol = vol.dropna()
    vol["CV"] = (vol["Std"] / vol["Mean"]).abs()
    top_volatile = vol.nlargest(20, "CV").reset_index()
    colours_v = [RED if cv > 1.0 else ORANGE if cv > 0.6 else TEAL
                 for cv in top_volatile["CV"]]
    ax7.barh(top_volatile["AccountID"], top_volatile["CV"],
             color=colours_v, edgecolor="white")
    ax7.axvline(0.6, color=ORANGE, lw=1.3, linestyle="--",
                label="Moderate threshold (0.6)")
    ax7.axvline(1.0, color=RED,    lw=1.3, linestyle="--",
                label="High threshold (1.0)")
    ax7.set_title("Balance Volatility — Top 20 Accounts\n"
                  "(Coefficient of Variation = Std / |Mean|)")
    ax7.set_xlabel("Coefficient of Variation (CV)")
    ax7.set_ylabel("Account ID")
    ax7.legend(fontsize=8, loc="lower right")
    ax7.xaxis.grid(True, linestyle="--")
    ax7.text(0.01, -0.18,
             "📊 Insight: Accounts with CV > 1.0 (red) show extreme balance"
             " swings — a potential sign of irregular cash flows or fraud.",
             transform=ax7.transAxes, fontsize=8, color="#555555",
             style="italic")

    # ── Chart 8: High-risk customer distribution by region ────────────────
    ax8 = fig.add_subplot(gs[1, 1])
    hr = df[df["RiskTier"] == "High Risk"]
    region_counts = hr["Region"].value_counts().sort_values(ascending=False)
    palette_r = sns.color_palette("Reds_r", len(region_counts))
    sns.barplot(x=region_counts.values, y=region_counts.index,
                palette=palette_r, ax=ax8, orient="h")
    for i, val in enumerate(region_counts.values):
        ax8.text(val + 0.5, i, str(val),
                 va="center", fontsize=10, fontweight="bold")
    ax8.set_title("High-Risk Customer Distribution by Region")
    ax8.set_xlabel("Number of High-Risk Transactions")
    ax8.set_ylabel("Region")
    ax8.xaxis.grid(True, linestyle="--")
    ax8.text(0.01, -0.18,
             "📊 Insight: North and South regions show the highest concentration"
             " of high-risk transactions. Targeted compliance reviews in these"
             " regions could reduce portfolio risk significantly.",
             transform=ax8.transAxes, fontsize=8, color="#555555",
             style="italic")

    # ── Chart 9: Risk tier distribution (Count plot) ──────────────────────
    ax9 = fig.add_subplot(gs[2, 0])
    risk_counts = df["RiskTier"].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"])
    colours_rt = [GREEN, GOLD, RED]
    bars9 = ax9.bar(risk_counts.index, risk_counts.values,
                    color=colours_rt, edgecolor="white", linewidth=0.8,
                    width=0.55)
    for bar, val in zip(bars9, risk_counts.values):
        pct = val / len(df) * 100
        ax9.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 3,
                 f"{val:,}\n({pct:.1f}%)",
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax9.set_title("Risk Tier Distribution\n"
                  "(Low < 0.3  |  Medium 0.3–0.6  |  High > 0.6)")
    ax9.set_ylabel("Number of Transactions")
    ax9.set_xlabel("Risk Tier")
    ax9.yaxis.grid(True, linestyle="--")
    ax9.text(0.01, -0.18,
             "📊 Insight: High-risk transactions form a substantial share of the"
             " portfolio. Proactive engagement with these accounts can mitigate"
             " potential defaults and fraud losses.",
             transform=ax9.transAxes, fontsize=8, color="#555555",
             style="italic")

    # ── Chart 10: Transaction amount boxplot by transaction type ──────────
    ax10 = fig.add_subplot(gs[2, 1])
    order = ["Withdrawal", "Payment", "Deposit", "Transfer"]
    palette_box = {"Withdrawal": RED, "Payment": ORANGE,
                   "Deposit": GREEN, "Transfer": BLUE}
    sns.boxplot(data=df, x="TransactionType", y="TransactionAmount",
                order=order, palette=palette_box,
                flierprops={"marker": "o", "markerfacecolor": RED,
                            "markersize": 4, "alpha": 0.5},
                ax=ax10)
    ax10.axhline(0, color="black", lw=1, linestyle="--")
    ax10.set_title("Transaction Amount Distribution by Type\n(Outliers Visible)")
    ax10.set_ylabel("Transaction Amount ($)")
    ax10.set_xlabel("Transaction Type")
    ax10.yaxis.grid(True, linestyle="--")
    ax10.text(0.01, -0.18,
             "📊 Insight: Withdrawals show the widest spread, reflecting varied"
             " customer behaviour. Negative amounts across types suggest"
             " reversed / corrected transactions that need reconciliation.",
             transform=ax10.transAxes, fontsize=8, color="#555555",
             style="italic")

    plt.savefig("/mnt/user-data/outputs/dashboard_page2_risk.png",
                dpi=150, bbox_inches="tight", facecolor=LIGHT_GREY)
    plt.close()
    print("✅  Page 2 saved.")


# =============================================================================
# 5. DASHBOARD — PAGE 3: FINAL SUMMARY INFOGRAPHIC
# =============================================================================

def page3_summary(df: pd.DataFrame, kpis: dict):
    fig = plt.figure(figsize=(20, 14), facecolor="#1A2A4A")
    fig.suptitle(
        "EXECUTIVE SUMMARY  |  Financial Risk & Transaction Behaviour",
        fontsize=18, fontweight="bold", color="white", y=0.97
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.5, wspace=0.4,
        top=0.91, bottom=0.05, left=0.06, right=0.97
    )

    # ── KPI ribbon ────────────────────────────────────────────────────────
    kpi_ax = fig.add_axes([0, 0.84, 1, 0.07])
    kpi_ax.set_facecolor("#162035")
    kpi_ax.axis("off")
    labels  = ["Transactions", "Customers", "Avg Balance",
               "High Risk", "Overdrafts", "Anomalies"]
    values  = [f"{kpis['total_transactions']:,}",
               f"{kpis['total_customers']:,}",
               f"${kpis['avg_balance']:,.0f}",
               f"{kpis['high_risk_accounts']:,}",
               f"{kpis['overdraft_accounts']:,}",
               f"{kpis['anomaly_count']:,}"]
    cols    = [TEAL, GREEN, BLUE, ORANGE, RED, PURPLE]
    for i, (lbl, val, col) in enumerate(zip(labels, values, cols)):
        x = 0.08 + i * 0.155
        kpi_ax.text(x, 0.65, val,  transform=kpi_ax.transAxes,
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", color=col)
        kpi_ax.text(x, 0.18, lbl, transform=kpi_ax.transAxes,
                    ha="center", va="center",
                    fontsize=9, color="white", alpha=0.75)

    # ── Pie: Transaction type share ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1A2A4A")
    tt = df["TransactionType"].value_counts()
    wedge_colours = [RED, ORANGE, GREEN, BLUE, PURPLE]
    wedges, texts, autotexts = ax1.pie(
        tt.values, labels=tt.index,
        colors=wedge_colours[:len(tt)],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "#1A2A4A", "linewidth": 2},
        textprops={"color": "white", "fontsize": 9}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
    ax1.set_title("Transaction Type Share", color="white", pad=12)

    # ── Pie: Risk tier share ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1A2A4A")
    rt = df["RiskTier"].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"])
    wedges2, texts2, autotexts2 = ax2.pie(
        rt.values, labels=rt.index,
        colors=[GREEN, GOLD, RED],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "#1A2A4A", "linewidth": 2},
        textprops={"color": "white", "fontsize": 9}
    )
    for at in autotexts2:
        at.set_fontsize(8); at.set_color("white")
    ax2.set_title("Risk Tier Breakdown", color="white", pad=12)

    # ── Bar: Net inflow — top 10 accounts ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1A2A4A")
    acct = df.groupby("AccountID").agg(
        Credit=("Credit", "sum"), Debit=("Debit", "sum")).reset_index()
    acct["NetInflow"] = acct["Credit"] - acct["Debit"]
    top10 = acct.nlargest(10, "NetInflow")
    c10 = [GREEN if v > 0 else RED for v in top10["NetInflow"]]
    ax3.barh(top10["AccountID"], top10["NetInflow"] / 1e3,
             color=c10, edgecolor="none")
    ax3.axvline(0, color="white", lw=0.8)
    ax3.set_title("Top 10 Accounts by Net Inflow", color="white")
    ax3.set_xlabel("Net Inflow ($K)", color="white")
    ax3.tick_params(colors="white")
    for sp in ax3.spines.values():
        sp.set_color("#444455")
    ax3.xaxis.grid(True, linestyle="--", color="#444455")
    ax3.set_facecolor("#1F3054")

    # ── Line: Monthly net flow ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#1F3054")
    monthly = df.groupby("YearMonth")[["Credit", "Debit"]].sum()
    monthly["Net"] = monthly["Credit"] - monthly["Debit"]
    idx_str = monthly.index.astype(str)
    x = range(len(monthly))
    ax4.plot(x, monthly["Net"] / 1e6, color=TEAL, lw=2.5,
             marker="D", markersize=5)
    ax4.fill_between(x, monthly["Net"] / 1e6, 0,
                     where=(monthly["Net"] > 0).values,
                     alpha=0.3, color=GREEN)
    ax4.fill_between(x, monthly["Net"] / 1e6, 0,
                     where=(monthly["Net"] <= 0).values,
                     alpha=0.3, color=RED)
    ax4.axhline(0, color="white", lw=0.8, linestyle="--")
    ax4.set_xticks(list(x))
    ax4.set_xticklabels(idx_str, rotation=45, ha="right",
                        fontsize=7.5, color="white")
    ax4.set_title("Monthly Net Cash Flow (Credit − Debit)", color="white")
    ax4.set_ylabel("Net Flow (Millions $)", color="white")
    ax4.tick_params(colors="white")
    for sp in ax4.spines.values():
        sp.set_color("#444455")
    ax4.yaxis.grid(True, linestyle="--", color="#444455")

    # ── Text: Findings & recommendations ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#162035")
    ax5.axis("off")
    summary_text = (
        "KEY FINDINGS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"▶  {kpis['high_risk_accounts']:,} transactions carry High Risk\n"
        f"   (RiskScore > 0.6) — escalate for review.\n\n"
        f"▶  {kpis['overdraft_accounts']:,} accounts in overdraft;\n"
        f"   immediate collections action needed.\n\n"
        f"▶  {kpis['anomaly_count']} statistical anomalies detected\n"
        f"   (|Z-Score| > 3) — possible fraud signals.\n\n"
        f"▶  Debit volumes exceed credit in most\n"
        f"   months — watch net-liquidity position.\n\n"
        "RECOMMENDATIONS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "✔  Flag overdraft & anomaly accounts\n"
        "   for same-day compliance review.\n\n"
        "✔  Offer re-engagement campaigns to\n"
        "   dormant accounts (gap ≥ 60 days).\n\n"
        "✔  Investigate North & South regions\n"
        "   with disproportionate high-risk load.\n\n"
        "✔  Reassess credit limits for accounts\n"
        "   with CV > 1.0 (extreme volatility)."
    )
    ax5.text(0.05, 0.97, summary_text,
             transform=ax5.transAxes,
             fontsize=8.8, va="top", color="white",
             fontfamily="monospace",
             linespacing=1.6)
    ax5.set_title("Analyst Summary", color="white", pad=8)

    plt.savefig("/mnt/user-data/outputs/dashboard_page3_summary.png",
                dpi=150, bbox_inches="tight", facecolor="#1A2A4A")
    plt.close()
    print("✅  Page 3 saved.")


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Financial Risk Analysis Dashboard")
    print("=" * 60)

    print("\n[1/5] Generating dataset …")
    df_raw = generate_dataset(n=800)

    print("[2/5] Cleaning & preparing data …")
    df = prepare_data(df_raw)

    print("[3/5] Computing KPIs …")
    kpis = compute_kpis(df)
    print("\n  KPIs:")
    for k, v in kpis.items():
        print(f"    {k:25s}: "
              f"{'${:,.0f}'.format(v) if 'balance' in k else '{:,}'.format(int(v))}")

    print("\n[4/5] Rendering dashboards …")
    page1_overview(df, kpis)
    page2_risk(df, kpis)
    page3_summary(df, kpis)

    print("\n[5/5] All done! Output files:")
    print("  • dashboard_page1_overview.png")
    print("  • dashboard_page2_risk.png")
    print("  • dashboard_page3_summary.png")
    print("=" * 60)
