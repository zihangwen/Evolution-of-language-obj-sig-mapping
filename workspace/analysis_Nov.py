# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

BASE_PATH = Path("/home/zihangw/EvoComm")
plt.rcParams.update({'font.size': 10})

# sns.color_palette("tab10")

# %% ----- ----- ----- ----- ----- fixed population invasion analysis ----- ----- ----- ----- ----- %% #
combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_fix_popsize.csv"
graph_info_path = BASE_PATH / "results_invade_combined" / "invade_param_demes_fix_popsize_info.csv"

df_result = pd.read_csv(combined_name, sep="\t")
df_info = pd.read_csv(graph_info_path, sep="\t")
df_all = pd.merge(df_result, df_info, on="graph_name")

# %% df select
df_wm = df_all[df_all["graph_base"] == "wm"]
df_graphs = df_all[df_all["graph_base"] != "wm"]

# %% fixation probability vs algebraic_connectivity for num_demes = 20
df_select = df_graphs[df_graphs["num_demes"] == 20]
sns.lineplot(data=df_select, x="algebraic_connectivity", y="pfix", markers=True, dashes=False, palette="Set1")
plt.axhline(y=df_wm["pfix"].mean(), color='black', linestyle='--', label='wm mean pfix')
plt.legend()
plt.title("Fixation Probability vs Algebraic Connectivity (num_demes=20)")
plt.show()

sns.lineplot(data=df_select, x="num_edge_added", y="pfix", markers=True, dashes=False, palette="Set1")
plt.axhline(y=df_wm["pfix"].mean(), color='black', linestyle='--', label='wm mean pfix')
plt.legend()
plt.title("Fixation Probability vs num_edge_added (num_demes=20)")
plt.show()

# %%
sns.lineplot(data=df_graphs, x="algebraic_connectivity", y="pfix", hue = "num_demes",markers=True, dashes=False, palette="Set1")
plt.axhline(y=df_wm["pfix"].mean(), color='black', linestyle='--', label='wm mean pfix')
plt.legend()
plt.title("Fixation Probability vs Algebraic Connectivity (num_demes=20)")
plt.show()

# %% time to fixation vs algebraic_connectivity for num_demes = 20
df_select = df_graphs[df_graphs["num_demes"] == 20]
sns.lineplot(data=df_select, x="algebraic_connectivity", y="fixation_time", markers=True, dashes=False, palette="Set1")
plt.axhline(y=df_wm["fixation_time"].mean(), color='black', linestyle='--', label='wm mean fixation_time')
plt.legend()
plt.title("Time to Fixation vs Algebraic Connectivity (num_demes=20)")
plt.show()

sns.lineplot(data=df_select, x="num_edge_added", y="fixation_time", markers=True, dashes=False, palette="Set1")
plt.axhline(y=df_wm["fixation_time"].mean(), color='black', linestyle='--', label='wm mean fixation_time')
plt.legend()
plt.title("Time to Fixation vs num_edge_added (num_demes=20)")
plt.show()

# %%
sns.lineplot(data=df_graphs, x="algebraic_connectivity", y="fixation_time", hue = "num_demes",markers=True, dashes=False, palette="Set1")
plt.title("Time to Fixation vs Algebraic Connectivity")
plt.show()


# %% ----- ----- ----- ----- ----- fixed population evolve analysis ----- ----- ----- ----- ----- %% #
combined_name_evolve = BASE_PATH / "results_evolve_combined" / "evolve_param_demes_fix_popsize.csv"

df_evolve = pd.read_csv(combined_name_evolve, sep="\t")
df_evolve_all = pd.merge(df_evolve, df_info, on="graph_name")
df_evolve_wm = df_evolve_all[df_evolve_all["graph_base"] == "wm"]
df_evolve_graphs = df_evolve_all[df_evolve_all["graph_base"] != "wm"]

# %% final_num_lang vs algebraic_connectivity for num_demes = 20
df_select = df_evolve_graphs[df_evolve_graphs["num_demes"] == 20]
sns.lineplot(data=df_select, x="algebraic_connectivity", y="final_num_lang", markers=True, dashes=False, palette="Set1")
plt.axhline(y=df_evolve_wm["final_num_lang"].mean(), color='black', linestyle='--', label='wm mean final_num_lang')
plt.legend()
plt.title("Final Number of Languages vs Algebraic Connectivity (num_demes=20)")
plt.show()

# %%
sns.lineplot(data=df_evolve_graphs, x="algebraic_connectivity", y="final_num_lang", hue = "num_demes",markers=True, dashes=False, palette="Set1")
plt.axhline(y=df_evolve_wm["final_num_lang"].mean(), color='black', linestyle='--', label='wm mean final_num_lang')
plt.legend()
plt.title("Final Number of Languages vs Algebraic Connectivity")
plt.show()

# %% time to final_num_lang vs algebraic_connectivity
sns.lineplot(data=df_evolve_graphs, x="algebraic_connectivity", y="final_num_lang_time", hue="num_demes", markers=True, dashes=False, palette="Set1")
plt.axhline(y=df_evolve_wm["final_num_lang_time"].mean(), color='black', linestyle='--', label='wm mean final_num_lang_time')
plt.legend()
plt.title("Time to Final Number of Languages vs algebraic_connectivity")
plt.show()


# %% ----- ----- ----- ----- fixed deme size or num demes invasion analysis ----- ----- ----- ----- %% #
combined_name = BASE_PATH / "results_invade_combined" / "invade_param_demes_multi_ns.csv"
graph_info_path = BASE_PATH / "results_invade_combined" / "invade_param_demes_multi_ns_info.csv"
df_result = pd.read_csv(combined_name, sep="\t")
df_info = pd.read_csv(graph_info_path, sep="\t")
df_all = pd.merge(df_result, df_info, on="graph_path")


# %% pfix vs num_demes for deme_size = 10 and num_edge_added = 0, 5
df_select = df_all[(df_all["deme_size"] == 10) & (df_all["num_edge_added"].isin([0, 5]))]
# df_base = df_all[(df_all["deme_size"] == 10) & (df_all["num_edge_added"] == 0)]
sns.lineplot(data=df_select, x="num_demes", y="pfix", markers=True, dashes=False, palette="Set1")
# plt.axhline(y=df_base["pfix"].mean(), color='black', linestyle='--', label='base mean pfix')
plt.title("Fixation Probability vs Number of Demes (deme_size=10)")
plt.show()

# %%
df_select = df_all[(df_all["num_edge_added"].isin([0, 5]))]
# df_base = df_all[(df_all["deme_size"] == 10) & (df_all["num_edge_added"] == 0)]
sns.lineplot(data=df_select, x="num_demes", y="pfix", hue = "deme_size", markers=True, dashes=False, palette="Set1")
plt.title("Fixation Probability vs Number of Demes")
plt.show()

# plt.axhline(y=df_base["pfix"].mean(), color='black', linestyle='--', label='base mean pfix')
# %%
df_select = df_all[(df_all["deme_size"] == 10) & (df_all["num_edge_added"].isin([0, 5]))]
# df_base = df_all[(df_all["deme_size"] == 10) & (df_all["num_edge_added"] == 0)]
sns.lineplot(data=df_select, x="num_demes", y="fixation_time", markers=True, dashes=False, palette="Set1")
# plt.axhline(y=df_base["fixation_time"].mean(), color='black', linestyle='--', label='base mean fixation_time')
plt.title("Time to Fixation vs Number of Demes (deme_size=10)")
plt.show()

# %%
df_select = df_all[(df_all["num_edge_added"].isin([0, 5]))]
# df_base = df_all[(df_all["deme_size"] == 10) & (df_all["num_edge_added"] == 0)]
sns.lineplot(data=df_select, x="num_demes", y="fixation_time", hue = "deme_size", markers=True, dashes=False, palette="Set1")
plt.title("Time to Fixation vs Number of Demes")
plt.show()

# %% pfix vs deme_size for num_demes = 20 and num_edge_added = 0, 5
df_select = df_all[(df_all["num_demes"] == 20) & (df_all["num_edge_added"].isin([0, 5]))]
# df_base = df_all[(df_all["num_demes"] == 20) & (df_all["num_edge_added"] == 0)]
sns.lineplot(data=df_select, x="deme_size", y="pfix", markers=True, dashes=False, palette="Set1")
# plt.axhline(y=df_base["pfix"].mean(), color='black', linestyle='--', label='base mean pfix')
plt.title("Fixation Probability vs Deme Size (num_demes=20)")
plt.show()

# %%
df_select = df_all[(df_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_select, x="deme_size", y="pfix", hue = "num_demes", markers=True, dashes=False, palette="Set1")
plt.title("Fixation Probability vs Deme Size")
plt.show()

# %%
df_select = df_all[(df_all["num_demes"] == 20) & (df_all["num_edge_added"].isin([0, 5]))]
# df_base = df_all[(df_all["num_demes"] == 20) & (df_all["num_edge_added"] == 0)]
sns.lineplot(data=df_select, x="deme_size", y="fixation_time", markers=True, dashes=False, palette="Set1")
# plt.axhline(y=df_base["fixation_time"].mean(), color='black', linestyle='--', label='base mean fixation_time')
plt.title("Time to Fixation vs Deme Size (num_demes=20)")
plt.show()

# %%
df_select = df_all[(df_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_select, x="deme_size", y="fixation_time", hue = "num_demes", markers=True, dashes=False, palette="Set1")
plt.title("Time to Fixation vs Deme Size")
plt.show()

# %% ----- ----- ----- ----- fixed deme size or num demes evolve analysis ----- ----- ----- ----- %% #
combined_name_evolve = BASE_PATH / "results_evolve_combined" / "evolve_param_demes_multi_ns.csv"
df_evolve = pd.read_csv(combined_name_evolve, sep="\t")
df_evolve_all = pd.merge(df_evolve, df_info, on="graph_path")

# set(df_info["graph_path"]) - set(df_evolve["graph_path"])

# %% final_num_lang vs num_demes for deme_size = 10 and num_edge_added = 5
df_evolve_select = df_evolve_all[(df_evolve_all["deme_size"] == 10) & (df_evolve_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_evolve_select, x="num_demes", y="final_num_lang", markers=True, dashes=False, palette="Set1")
plt.title("Final Number of Languages vs Number of Demes (deme_size=10)")
plt.show()

# %%
df_evolve_select = df_evolve_all[(df_evolve_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_evolve_select, x="num_demes", y="final_num_lang", hue = "deme_size", markers=True, dashes=False, palette="Set1")
plt.title("Final Number of Languages vs Number of Demes")
plt.show()

# %% time to final_num_lang vs num_demes for deme_size = 10 and num_edge_added = 5
df_evolve_select = df_evolve_all[(df_evolve_all["deme_size"] == 10) & (df_evolve_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_evolve_select, x="num_demes", y="final_num_lang_time", markers=True, dashes=False, palette="Set1")
plt.title("Time to Final Number of Languages vs Number of Demes (deme_size=10)")
plt.show()

# %%
df_evolve_select = df_evolve_all[(df_evolve_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_evolve_select, x="num_demes", y="final_num_lang_time", hue = "deme_size", markers=True, dashes=False, palette="Set1")
plt.title("Time to Final Number of Languages vs Number of Demes")
plt.show()

# %% final_num_lang vs deme_size for num_demes = 20 and num_edge_added = 5
df_evolve_select = df_evolve_all[(df_evolve_all["num_demes"] == 20) & (df_evolve_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_evolve_select, x="deme_size", y="final_num_lang", markers=True, dashes=False, palette="Set1")
plt.title("Final Number of Languages vs Deme Size (num_demes=20)")
plt.show()

# %%
df_evolve_select = df_evolve_all[(df_evolve_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_evolve_select, x="deme_size", y="final_num_lang", hue = "num_demes", markers=True, dashes=False, palette="Set1")
plt.title("Final Number of Languages vs Deme Size")
plt.show()

# %% time to final_num_lang vs deme_size for num_demes = 20 and num_edge_added = 5
df_evolve_select = df_evolve_all[(df_evolve_all["num_demes"] == 20) & (df_evolve_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_evolve_select, x="deme_size", y="final_num_lang_time", markers=True, dashes=False, palette="Set1")
plt.title("Time to Final Number of Languages vs Deme Size (num_demes=20)")
plt.show()

# %%
df_evolve_select = df_evolve_all[(df_evolve_all["num_edge_added"].isin([0, 5]))]
sns.lineplot(data=df_evolve_select, x="deme_size", y="final_num_lang_time", hue = "num_demes", markers=True, dashes=False, palette="Set1")


# %%
