#%%
import time
from FileHandling import *
import numpy as np
from matplotlib.patches import Circle
from pathlib import Path
import seaborn as sns
import pingouin as pg
import warnings
from tabulate import tabulate
warnings.filterwarnings('ignore')
from joblib import dump,load
print("Packages are imported")

# %% run
summary = iterate_dataset(
    subjects=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    cursors=["Head", "MM"],
    selections=["Dwell", "Score"],
    repetitions=range(2, 10),
)
summary.to_pickle('summary.pkl')
# %%
summary = pd.read_pickle("summary.pkl")

#%%
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
by_subjects_success_only = success_only.groupby(
    [success_only.subject, success_only.cursor, success_only.selection]
).mean()

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
custom_params = {"axes.spines.right": True, "axes.spines.top": True}
sns.set_theme(
    context="paper",  # 매체: paper, talk, poster
    style="whitegrid",  # 기본 내장 테마
    palette="muted",  # 그래프 색
    font_scale=3,  # 글꼴 크기
    rc=custom_params,
)  # 그래프 세부 사항
def format_label(label):
    return label.replace("_", " ").title()

for c in ["total_time", "success", "target_entries", "contact_time", "selection_time"]:

    if c  in  ["selection_time","total_time",'contact_time']:
        dataset = by_subjects_success_only.copy()
    else:
        dataset = by_subjects.copy()

    g = sns.catplot(
        data=dataset,
        x="cursor",
        y=c,
        hue="selection",
        kind="box",
        showfliers=True,
        showmeans=True,
        meanprops={
            "marker": "x",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "10",
        },
        height=10,
        aspect=0.8,
        # legend_out=False,
    )
    for ax in g.axes.flat:
        ax.set_xlabel(format_label(ax.get_xlabel()))
        ax.set_ylabel(format_label(ax.get_ylabel()))
    g.set_titles(col_template="{col_name}".title())
    plt.title(format_label(c))
    # plt.tight_layout()
    plt.show()
    # plt.savefig("Plots/" + c + ".pdf")
# %% RM anova

for c in [
    # "total_time",
    #   "success", 
          "target_entries",
            # "contact_time", "selection_time"
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
    # Parameters
    tail_length = 10  # Number of points in the tail
    if summary.cursor[i] == "MM":
        useHead = True
    else:
        useHead = False
    # Initialize figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(df["horizontal_offset"].min() - 2, df["horizontal_offset"].max() + 2)
    ax.set_ylim(df["vertical_offset"].min() - 2, df["vertical_offset"].max() + 2)
    ax.set_aspect("equal")
    # (point,) = ax.plot([], [], "bo", label="Moving Point")  # Moving point
    tail = ax.scatter([], [], c="red", s=20, alpha=0.5, label="Tail")  # Tail
    scat = ax.scatter([], [], c="blue", s=50)  # Points only
    if useHead:
        head = ax.scatter([], [], c="green", s=30, alpha=0.5)

    # Initialization function
    def init():
        scat.set_offsets([])  # Empty the scatter
        tail.set_offsets([])  # Empty the scatter
        if useHead:
            head.set_offsets([])
        return (scat,)

    # Update function
    def update(frame):
        # Get all points up to the current frame
        circle = Circle(
            (0, 0),
            radius=1.5,
            edgecolor="green",
            facecolor="none",
            linestyle="--",
            linewidth=2,
        )
        plt.gca().add_patch(circle)
        x_data = df["horizontal_offset"][frame : frame + 1]
        y_data = df["vertical_offset"][frame : frame + 1]
        start = max(0, frame - tail_length)
        tail_x = df["horizontal_offset"][start : frame + 1]
        tail_y = df["vertical_offset"][start : frame + 1]
        if useHead:
            x = df["head_horizontal_offset"][frame : frame + 1]
            y = df["head_vertical_offset"][frame : frame + 1]
            head.set_offsets(list(zip(x, y)))
        scat.set_offsets(list(zip(x_data, y_data)))
        tail.set_offsets(list(zip(tail_x, tail_y)))
        plt.title(f"{summary.cursor[i]}_{summary.selection[i]}")

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
