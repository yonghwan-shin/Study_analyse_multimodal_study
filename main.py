# %%
import time
from FileHandling import *
import numpy as np
from matplotlib.patches import Circle
from pathlib import Path
import seaborn as sns
import pingouin as pg
import warnings
from tabulate import tabulate

warnings.filterwarnings("ignore")
from joblib import dump, load

print("Packages are imported")
# %%
app_summary = pd.DataFrame(
    columns=[
        "subject_num",
        "condition",
        "repetition",
        "duration",
        "walklength",
        "walkspeed",
        "finalspeed",
    ]
)
for s in range(8):
    for i in range(12):
        try:
            d = read_app_data(s, "head", i)
            duration = d.timestamp.values[-1]
            d["movement"] = (
                d.head_origin_x.diff(1) ** 2
                + d.head_origin_y.diff(1) ** 2
                + d.head_origin_z.diff(1) ** 2
            ).apply(math.sqrt)
            walklength = d.movement.sum()

            trial_summary = {
                "subject_num": s,
                "condition": "head",
                "repetition": i,
                "duration": duration,
                "walklength": walklength,
                "walkspeed": walklength / duration,
                "finalspeed": d.movement[-60:].sum(),
            }
            app_summary.loc[len(app_summary)] = trial_summary

            d = read_app_data(s, "mm", i)
            duration = d.timestamp.values[-1]
            d["movement"] = (
                d.head_origin_x.diff(1) ** 2
                + d.head_origin_y.diff(1) ** 2
                + d.head_origin_z.diff(1) ** 2
            ).apply(math.sqrt)
            walklength = d.movement.sum()
            trial_summary = {
                "subject_num": s,
                "condition": "MM",
                "repetition": i,
                "duration": duration,
                "walklength": walklength,
                "walkspeed": walklength / duration,
                "finalspeed": d.movement[-60:].sum() * 2,
            }
            app_summary.loc[len(app_summary)] = trial_summary

        except Exception as e:
            print(e)
# %%
app_summary["stop"] = app_summary["finalspeed"] < 0.4
by_subject_app = app_summary.groupby(
    [app_summary.subject_num, app_summary.condition]
).mean()
b_app = pd.DataFrame(by_subject_app.to_records(), index=by_subject_app.index)

# %%


# %%
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    palette="muted",  # 그래프 색
    # palette='Set3',
    font_scale=3,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
bys = app_summary.groupby([app_summary.subject_num, app_summary.condition]).mean()
bys = pd.DataFrame(bys.to_records(), index=bys.index)
bys["condition"] = bys["condition"].map({"MM": "Combined", "head": "Baseline"})

import starbars

for c in ["duration", "finalspeed"]:
    fig, ax = plt.subplots(figsize=(6, 9))
    g = sns.boxplot(
        data=bys,
        x="condition",
        y=c,
        #    hue='condition',
        # showfliers=showfliers,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "lightgrey",
            "markeredgecolor": "black",
            "markersize": "10",
        },
        linewidth=2,
        fill=False,
        showcaps=False,
        width=0.4,
        ax=ax,
        legend=False,
        dodge=True,
        # gap=0.1
    )
    ax.set_xlabel("")

    if c == "duration":

        annotations = [("Combined", "Baseline", 0.001)]
        starbars.draw_annotation(annotations, fontsize=15)
        plt.title("Selection Time")
        ax.set(ylim=(0, 13.5))

        ax.set_ylabel("Selection Time (s)")
    else:
        annotations = [("Combined", "Baseline", 0.001)]
        starbars.draw_annotation(annotations, fontsize=15)
        plt.title("Walking Speed")
        ax.set(ylim=(0, 2.3))

        ax.set_ylabel("Walking Speed (m/s)")
    sns.despine(top=True, right=True, left=True, bottom=False)
    plt.tight_layout()
    # plt.savefig("final_speed.png")
    # sns.boxplot(data=bys,x='condition',y='duration')
    plt.savefig(c + ".pdf")
    # plt.show()
# %%

s = pd.read_csv("subjective_first.csv")
s["overall"] = s[
    [
        "Mental",
        "Physical",
        "Temporal",
        "Performance",
        "Effort",
        "Frustration",
    ]
].sum(axis=1)
for c in [
    # 'Mental',
    #       'Physical',
    #       'Temporal','Performance','Effort','Frustration',
    "Borg",
    "overall",
]:
    g = sns.catplot(data=s, x="Cursor", y=c, hue="Selection", kind="box", dodge=True)
    plt.title(c)
    plt.show()

    dataset = s.copy()
    dataset["Cursor"] = dataset["Cursor"].map({"Head": 0, "MM": 1})
    dataset["Selection"] = dataset["Selection"].map({"Dwell": 0, "Score": 1})
    dataset.loc[:, "Cursor"] = dataset["Cursor"].astype(int)
    dataset.loc[:, "Selection"] = dataset["Selection"].astype(int)
    aov = pg.rm_anova(
        dv=c,
        within=["Cursor", "Selection"],
        subject="Subject",
        data=dataset,
        detailed=True,
        effsize="ng2",
        correction=True,
    )
    aov.round(3)
    print(
        c.upper(),
    )
    pg.print_table(aov)

    # posthoc = pg.pairwise_tests(
    #     dv=c,
    #     within=["Cursor", "Selection"],
    #     subject="Subject",
    #     data=dataset,
    #     padjust="bonferroni",
    # )  # Adjust for multiple comparisons
    # from tabulate import tabulate

    # print(tabulate(posthoc, headers="keys", tablefmt="grid"))

# %% run
summary = iterate_dataset(
    subjects=[3, 4, 5],
    cursors=["Head", "MM"],
    selections=["Dwell", "Score"],
    repetitions=[0, 1],
)
summary.to_pickle("summary_stand.pkl")
# %%
summary = iterate_dataset(
    subjects=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    cursors=["Head", "MM"],
    selections=["Dwell", "Score"],
    repetitions=range(2, 10),
)
summary.to_pickle("summary.pkl")
# %%

summary["y_range"] = summary["data"].apply(
    lambda x: x.head_origin_y.max() - x.head_origin_y.min()
)
# %%

activation_thresholds = np.linspace(6, 12, num=7)  # example range
max_velocities = np.linspace(6, 12, num=7)  # deg/sec
for thresh in activation_thresholds:
    for vmax in max_velocities:

        print(f"Running analysis for threshold: {thresh}, vmax: {vmax}")
        summary["success" + str(thresh) + "_" + str(vmax)] = summary.apply(
            cap_velocity_analysis, args=(thresh, vmax), axis=1, result_type="expand"
        )
        print(thresh, vmax, summary["success" + str(thresh) + "_" + str(vmax)].mean())
# summary['sucess'+str(thresh)+'_'+str(vmax)] = summary.apply(cap_velocity_analysis, axis=1, result_type="expand")

# %%

success_cols = [col for col in summary.columns if col.startswith("success")]
df = summary[success_cols].mean().rename("success").reset_index()


# Rename 'index' to something meaningful
df = df.rename(columns={"index": "param"})

# Extract threshold and vmax using regex
split_cols = df["param"].str.split("_", expand=True)

# Remove "success" prefix and convert
df["thresh"] = split_cols[0].str.replace("success", "", regex=False).astype(float)
df["vmax"] = split_cols[1].astype(float)

print(df)
# %%
summary = pd.read_pickle("summary.pkl")
summary[["total_time", "contact_time", "selection_time"]] = summary.apply(
    time_analysis, axis=1, result_type="expand"
)
summary[["curv_idx", "path_length", "straight_dist", "avg_speed"]] = summary.apply(
    curvature_analysis, axis=1, result_type="expand"
)

summary["target_entries"] = summary.apply(entries_analysis, axis=1)

# summary["max_size"] = summary.apply(maxsize_analysis, axis=1)
results = summary.drop(["data"], axis=1).dropna()
results = results[results.repetition > 0]


success_only = results[results.success == 1]

by_subjects = results.groupby(
    [results.subject, results.cursor, results.selection]
).mean()
b = pd.DataFrame(by_subjects.to_records(), index=by_subjects.index)
by_subjects_success_only = success_only.groupby(
    [success_only.subject, success_only.cursor, success_only.selection]
).mean()
bb = pd.DataFrame(
    by_subjects_success_only.to_records(), index=by_subjects_success_only.index
)


# %% Defining error trials

summary["error"] = ""


def filter_group(group):
    # Select numeric columns for percentile calculation.
    exclude_cols = [
        "subject",
        "target",
        "total_time",
    ]

    # Select numeric columns for percentile calculations
    numeric_cols = group.select_dtypes(include="number").columns.difference(
        exclude_cols
    )
    # Calculate the 95th percentile for each numeric column in this group
    thresholds = group[numeric_cols].quantile(0.95)

    # For each row, determine which numeric columns exceed the threshold
    def row_error(row):
        exceeded_cols = [col for col in numeric_cols if row[col] > thresholds[col]]
        return ",".join(exceeded_cols) if exceeded_cols else ""

    # Create a new column 'error' with the names of the columns that exceed thresholds
    group["error"] = group.apply(row_error, axis=1)
    return group


# Group by both condition columns and apply the filtering function to each group.
print(len(results))
filtered_df = results.groupby(["cursor", "selection"], group_keys=False).apply(
    filter_group
)
print(len(filtered_df))

# %%
import seaborn as sns
import starbars
from itertools import combinations
from scipy.stats import ttest_ind

custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    palette="muted",  # 그래프 색
    # palette='Set3',
    font_scale=3,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항


def format_label(label):
    return label.replace("_", " ").title()


from statannotations.Annotator import Annotator
from statannot import add_stat_annotation

for c in [
    # "total_time",
    # "success",
    "target_entries",
    # "contact_time",
    # "selection_time",
]:

    if c in ["selection_time", "total_time", "contact_time"]:
        dataset = by_subjects_success_only.copy()
    else:
        dataset = by_subjects.copy()
    dataset = pd.DataFrame(dataset.to_records(), index=dataset.index)
    dataset["selection"] = dataset["selection"].map(
        {"Dwell": "Scoring-Off", "Score": "Scoring-On"}
    )
    dataset["cursor"] = dataset["cursor"].map({"Head": "VelCap-Off", "MM": "VelCap-On"})

    conditions = [
        (dataset["selection"] == "Scoring-Off") & (dataset["cursor"] == "VelCap-Off"),
        (dataset["selection"] == "Scoring-Off") & (dataset["cursor"] == "VelCap-On"),
        (dataset["selection"] == "Scoring-On") & (dataset["cursor"] == "VelCap-Off"),
        (dataset["selection"] == "Scoring-On") & (dataset["cursor"] == "VelCap-On"),
    ]

    values = ["Baseline", "VelCap", "Scoring", "Combined"]

    dataset["condition"] = np.select(conditions, values, default="unknown")
    palette = {
        "Baseline": "#1f77b4",
        "VelCap": "#ff7f0e",
        "Scoring": "#2ca02c",
        "Combined": "#d62728",
    }
    x_order = ["Baseline", "VelCap", "Scoring", "Combined"]
    pairs = list(combinations(dataset["condition"].unique(), 2))
    dataset["success"] = dataset["success"] * 100
    fig, ax = plt.subplots(figsize=(6, 9))
    showfliers = True
    if c == "contact_time":
        showfliers = False

    g = sns.boxplot(
        data=dataset,
        x="condition",
        y=c,
        # hue="cursor",
        showfliers=showfliers,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "lightgrey",
            "markeredgecolor": "black",
            "markersize": "10",
        },
        linewidth=2,
        order=x_order,
        fill=False,
        showcaps=False,
        width=0.4,
        ax=ax,
        legend=False,
        # dodge=True,
        # gap=0.1,
        # palette="muted",
        # height=10,
        # aspect=0.8,
        # legend_out=False,
    )
    results = []
    for cond1, cond2 in pairs:
        group1 = dataset[dataset["condition"] == cond1][c]
        group2 = dataset[dataset["condition"] == cond2][c]

        stat, p = ttest_ind(
            group1, group2, equal_var=False
        )  # Welch's t-test recommended
        results.append({"cond1": cond1, "cond2": cond2, "t-stat": stat, "p-value": p})
    results = pd.DataFrame(results)
    results["p-value"] = results["p-value"].round(3)
    print(results)
    annotations = [
        (row["cond1"], row["cond2"], row["p-value"]) for _, row in results.iterrows()
    ]
    ax.set_xlabel("")
    if c != "success":
        starbars.draw_annotation(
            annotations, fontsize=13, bar_gap=0.01, tip_length=0.01, ns_show=False
        )
    if c == "success":
        plt.title("Success Rate")
        starbars.draw_annotation(
            annotations, fontsize=13, bar_gap=-0.01, tip_length=0.01, ns_show=False
        )
        ax.set(ylim=(0, 125))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Success Rate (%)")

    if c == "selection_time":
        plt.title("Selection Time")
        ax.set(ylim=(0, 5))
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_ylabel("Selection Time (s)")
    if c == "contact_time":
        plt.title("Contact Time")
        ax.set(ylim=(0, 1.6))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        ax.set_ylabel("Contact Time (s)")

    if c == "target_entries":
        plt.title("Target Entries")
        ax.set(ylim=(0, 11))
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_ylabel("Count")
    plt.xticks(rotation=90)
    # plt.title(format_label(c))
    # plt.legend(loc="lower center", ncol=len(dataset.columns))
    # plt.tight_layout()
    # plt.legend(fontsize=20)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    # plt.show()
    plt.savefig("Plots/" + c + ".pdf")
# %% RM anova

for c in [
    # "total_time",
    #   "success",
    #   "target_entries",
    "contact_time",
    "selection_time",
]:
    dataset = summary.dropna(subset=[c])
    if c in ["selection_time", "total_time", "contact_time"]:
        dataset = success_only.copy()
    # dataset = success_only.dropna(subset=[c])
    # dataset=success_only.copy()
    dataset["cursor"] = dataset["cursor"].map({"Head": 0, "MM": 1})
    dataset["selection"] = dataset["selection"].map({"Dwell": 0, "Score": 1})
    dataset.loc[:, "cursor"] = dataset["cursor"].astype(int)
    dataset.loc[:, "selection"] = dataset["selection"].astype(int)
    aov = pg.rm_anova(
        dv=c,
        within=["cursor", "selection"],
        subject="subject",
        data=dataset,
        detailed=True,
        effsize="ng2",
        correction=True,
    )
    aov.round(3)
    print(
        c.upper(),
    )
    pg.print_table(aov)

    posthoc = pg.pairwise_tests(
        dv=c,
        within=["cursor", "selection"],
        subject="subject",
        data=dataset,
        padjust="bonferroni",
    )  # Adjust for multiple comparisons
    from tabulate import tabulate

    print(tabulate(posthoc, headers="keys", tablefmt="grid"))


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Compute means & standard errors for each condition
df_summary = (
    dataset.groupby(["cursor", "selection"])["selection_time"]
    .agg(["mean", "sem"])
    .reset_index()
)

# Create the interaction plot
plt.figure(figsize=(8, 6))
sns.pointplot(
    data=df_summary,
    x="cursor",
    y="mean",
    hue="selection",
    markers=["o", "s"],
    capsize=0.1,
    errwidth=1.2,
    dodge=True,
)

# Labels & formatting
plt.title("Interaction Effect of Cursor & Selection on Selection Time", fontsize=14)
plt.xlabel("Multimodal Cursor (0 = Off, 1 = On)", fontsize=12)
plt.ylabel("Mean Selection Time (s)", fontsize=12)
plt.xticks([0, 1], ["Off", "On"])
plt.legend(title="Scoring Method", labels=["Off", "On"])
plt.grid(True)

# Show plot
plt.show()
# %%

# %% draw gifs


import matplotlib.pyplot as plt


from matplotlib.animation import FuncAnimation, PillowWriter

# Example DataFrame
# Replace this with your actual data
# data = {"x": [0, 1, 2, 3, 4, 5, 6], "y": [0, 1, 0, -1, 0, 1, 0]}
# df = pd.DataFrame(data)
for i in range(len(summary)):
    # for i in [1]:
    df = summary.data[i]
    if summary.cursor[i] == "MM" and summary.selection[i] == "Score":
        continue
    if summary.cursor[i] == "Head" and summary.selection[i] == "Dwell":
        continue

    # if summary.subject[i]==0 and summary.cursor[i]  =='MM' and summary.selection[i] == 'Dwell' and summary.target[i] == 8 and summary.repetition[i] ==4:
    #     print("ok")
    if (
        summary.subject[i] == 0
        and summary.cursor[i] == "Head"
        and summary.selection[i] == "Score"
        and summary.target[i] == 1
        and summary.repetition[i] == 3
    ):
        print("ok")
    else:
        continue

    # Parameters
    tail_length = 5  # Number of points in the tail
    if summary.cursor[i] == "MM":
        useHead = True
    else:
        useHead = False

    # Initialize figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(df["horizontal_offset"].min() - 6, df["horizontal_offset"].max() + 6)
    ax.set_ylim(df["vertical_offset"].min() - 6, df["vertical_offset"].max() + 6)
    ax.set_aspect("equal")
    # (point,) = ax.plot([], [], "bo", label="Moving Point")  # Moving point
    tail = ax.scatter(
        [],
        [],
        c="red",
        s=20,
        alpha=0.3,
    )  # Tail
    scat = ax.scatter([], [], c="blue", s=50, label="Cursor")  # Points only
    # eye = ax.scatter([], [], c="red", s=1500,alpha=0.1,label='Eye')  # Points only
    if useHead:
        head = ax.scatter([], [], c="green", s=30, alpha=0.6, label="Head")

    # Initialization function
    def init():
        scat.set_offsets([])  # Empty the scatter
        tail.set_offsets([])  # Empty the scatter
        # eye.set_offsets([])
        if useHead:
            head.set_offsets([])
        return (scat,)

    # Update function
    def update(frame):
        # Get all points up to the current frame
        circle = Circle(
            (0, 0),
            radius=1.5,
            edgecolor="black",
            facecolor="none",
            linestyle="-",
            linewidth=2,
        )
        plt.gca().add_patch(circle)
        if summary.selection[i] == "Score":

            circle1 = Circle(
                (4.5, 0),
                radius=1.5,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
                linewidth=2,
            )
            plt.gca().add_patch(circle1)
            circle2 = Circle(
                (0, 4.5),
                radius=1.5,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
                linewidth=2,
            )
            plt.gca().add_patch(circle2)
            circle3 = Circle(
                (0, -4.5),
                radius=1.5,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
                linewidth=2,
            )
            plt.gca().add_patch(circle3)
            circle4 = Circle(
                (-4.5, 0),
                radius=1.5,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
                linewidth=2,
            )
            plt.gca().add_patch(circle4)
        x_data = df["horizontal_offset"][frame : frame + 1]
        y_data = df["vertical_offset"][frame : frame + 1]
        start = max(0, frame - tail_length)
        tail_x = df["horizontal_offset"][start : frame + 1]
        tail_y = df["vertical_offset"][start : frame + 1]

        eyex = df["eyeRay_horizontal_offset"][frame : frame + 1]
        eyey = df["eyeRay_vertical_offset"][frame : frame + 1]
        # eye.set_offsets(list(zip(eyex, eyey)))
        if useHead:
            x = df["head_horizontal_offset"][frame : frame + 1]
            y = df["head_vertical_offset"][frame : frame + 1]
            head.set_offsets(list(zip(x, y)))
        scat.set_offsets(list(zip(x_data, y_data)))
        tail.set_offsets(list(zip(tail_x, tail_y)))
        # plt.title(f"{summary.cursor[i]}_{summary.selection[i]}")
        plt.xticks([], [])
        plt.yticks([], [])
        # plt.legend()
        plt.grid(False)
        # fig.legend(loc='outside upper right')
        # plt.scatter(x_data, y_data)
        # Update scatter plot with new points
        # scat.set_offsets(list(zip(x_data, y_data)))
        # return (scat,)

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(len(df)), interval=200)

    # Show the animation
    # plt.legend()
    # plt.show()
    directory = Path(f"gifs/{summary.subject[i]}")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    ani.save(
        f"gifs/{summary.subject[i]}/{summary.subject[i]}_{summary.cursor[i]}_{summary.selection[i]}_{summary.target[i]}_{summary.repetition[i]}_{summary.success[i]}_animation.gif",
        writer="imagemagick",
        fps=30,
    )
# %%
for i in range(len(summary)):
    # for i in [1]:
    df = summary.data[i]
    if summary.cursor[i] == "MM" and summary.selection[i] == "Score":
        continue
    if summary.cursor[i] == "Head" and summary.selection[i] == "Dwell":
        continue

    # if summary.subject[i]==0 and summary.cursor[i]  =='MM' and summary.selection[i] == 'Dwell' and summary.target[i] == 8 and summary.repetition[i] == 4:
    if (
        summary.subject[i] == 0
        and summary.cursor[i] == "Head"
        and summary.selection[i] == "Score"
        and summary.target[i] == 1
        and summary.repetition[i] == 3
    ):
        print("ok")
    else:
        continue

    fig, ax = plt.subplots(figsize=(12, 6))

    def velcap(d):
        if d > 9:
            return 0
        else:
            a = 6 + 6 * d / 9
            return a

    df["limit"] = df["EH_distance"].apply(velcap)

    xdata, ydata = [], []
    xdata2, ydata2 = [], []

    (ln,) = plt.plot([], [], "r")
    (ln2,) = plt.plot([], [], "b")

    def init():
        ax.set_xlim(0, 2)
        ax.set_ylim(0, df.EH_distance.max())
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Head-eye distance (deg)")
        ax.hlines(9, 0, df.EH_distance.max())
        return (ln,)

    def animate(frame):
        xdata.append(df["timestamp"][frame])
        ydata.append(df["EH_distance"][frame])
        ln.set_data(xdata, ydata)
        xdata2.append(df["timestamp"][frame])
        ydata2.append(df["limit"][frame])
        plt.tight_layout()
        return (ln,)

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        interval=200,
    )
    anim.save(
        f"gifs/{summary.subject[i]}/{summary.subject[i]}_{summary.cursor[i]}_{summary.selection[i]}_{summary.target[i]}_{summary.repetition[i]}_{summary.success[i]}_animation_graph.gif",
        writer="imagemagick",
        fps=30,
        dpi=100,
    )
    continue


# %%
for i in range(len(summary)):
    # for i in [1]:
    df = summary.data[i]
    if summary.cursor[i] == "MM" and summary.selection[i] == "Score":
        continue
    if summary.cursor[i] == "Head" and summary.selection[i] == "Dwell":
        continue

    # if summary.subject[i]==0 and summary.cursor[i]  =='MM' and summary.selection[i] == 'Dwell' and summary.target[i] == 8 and summary.repetition[i] == 4:
    if (
        summary.subject[i] == 0
        and summary.cursor[i] == "Head"
        and summary.selection[i] == "Score"
        and summary.target[i] == 1
        and summary.repetition[i] == 3
    ):
        print("ok")
    else:
        continue

    fig, ax = plt.subplots(figsize=(12, 6))

    xdata, ydata = [], []
    xdata2, ydata2 = [], []
    xdata3, ydata3 = [], []
    xdata4, ydata4 = [], []
    xdata5, ydata5 = [], []
    (ln,) = plt.plot([], [], "r", label="Target Score")
    (ln2,) = plt.plot([], [], "b", alpha=0.9, label="Distractor 1 Score")
    (ln3,) = plt.plot([], [], "b", alpha=0.7, label="Distractor 2 Score")
    (ln4,) = plt.plot([], [], "b", alpha=0.5, label="Distractor 3 Score")
    (ln5,) = plt.plot([], [], "b", alpha=0.3, label="Distractor 4 Score")

    def init():
        ax.set_xlim(0, df.timestamp.max())
        ax.set_ylim(0, 1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Score")

        return ln, ln2, ln3, ln4, ln5

    def animate(frame):
        xdata.append(df["timestamp"][frame])
        ydata.append(df["target_score"][frame])
        ln.set_data(xdata, ydata)

        xdata2.append(df["timestamp"][frame])
        ydata2.append(df["up_score"][frame])
        ln2.set_data(xdata2, ydata2)

        xdata3.append(df["timestamp"][frame])
        ydata3.append(df["right_score"][frame])
        ln3.set_data(xdata3, ydata3)

        xdata4.append(df["timestamp"][frame])
        ydata4.append(df["down_score"][frame])
        ln4.set_data(xdata4, ydata4)

        xdata5.append(df["timestamp"][frame])
        ydata5.append(df["left_score"][frame])
        ln5.set_data(xdata5, ydata5)

        plt.tight_layout()
        plt.legend()
        return ln, ln2, ln3, ln4, ln5

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        interval=200,
    )
    anim.save(
        f"gifs/{summary.subject[i]}/{summary.subject[i]}_{summary.cursor[i]}_{summary.selection[i]}_{summary.target[i]}_{summary.repetition[i]}_{summary.success[i]}_animation_graph.gif",
        writer="imagemagick",
        fps=30,
        dpi=100,
    )
    continue

# %%


def run_analysis():
    # Use a breakpoint in the code line below to debug your script.

    summary = iterate_dataset(
        subjects=[98],
        cursors=["Head", "MM"],
        selections=["Dwell", "Score"],
        repetitions=[0],
    )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    print(f"Analyzer started at, {current_time}")  # Press ⌘F8 to toggle the breakpoint.

    summary = run_analysis()

    current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    print(f"Analyzer finished at, {current_time}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# %%
data = pd.read_csv("csv_save_r1.csv")
by_subject = data.groupby(
    [data.subject, data.posture, data.selection, data.cursor]
).mean()
by_subject.to_csv("csv_save_r1_mean.csv")
# %%
by_subject = pd.read_csv("csv_save_r1_mean.csv")
by_subject[
    (by_subject.selection == "Click")
    & (by_subject.cursor == "Hand")
    & (by_subject.posture == "Treadmill")
].success.mean()
