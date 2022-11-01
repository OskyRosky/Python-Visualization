#######################################
#              Seaborn                # 
#######################################

# seaborn.pydata.org

# Seaborn is a library for making statistical graphics in Python.
# It builds on top of matplotlib and integrates closely with pandas data structures

# Import seaborn
import seaborn as sns


import matplotlib.pyplot as plt

# Apply the default theme
sns.set_theme()

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)
plt.show() #### ---->  matplotlib.pyplot.show() is requirred 

####### 

import seaborn as sns
df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")

# import matplotlib.pyplot as plt    --->  matplotlib.pyplot.show() is requirred 
plt.show() 

tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)
plt.show() 


###### A high-level API for statistical graphics

dots = sns.load_dataset("dots")
sns.relplot(
    data=dots, kind="line",
    x="time", y="firing_rate", col="align",
    hue="choice", size="coherence", style="choice",
    facet_kws=dict(sharex=False),
)
plt.show() 

# Statistical estimation

fmri = sns.load_dataset("fmri")
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", col="region",
    hue="event", style="event",
)
plt.show() 


sns.lmplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker")

plt.show() 


# Distributional representations

sns.displot(data=tips, x="total_bill", col="time", kde=False)
plt.show() 

sns.displot(data=tips, x="total_bill", col="time", kde=True)
plt.show() 

sns.displot(data=tips, kind="ecdf", x="total_bill", col="time", hue="smoker", rug=True)
plt.show() 

# Plots for categorical data

sns.catplot(data=tips, kind="swarm", x="day", y="total_bill", hue="smoker")
plt.show() 

sns.catplot(data=tips, kind="violin", x="day", y="total_bill", hue="smoker", split=True)
plt.show() 

sns.catplot(data=tips, kind="bar", x="day", y="total_bill", hue="smoker")
plt.show() 

penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")
plt.show() 

sns.pairplot(data=penguins, hue="species")
plt.show() 

### Lower-level tools for building figures

g = sns.PairGrid(penguins, hue="species", corner=True)
g.map_lower(sns.kdeplot, hue=None, levels=5, color=".2")
g.map_lower(sns.scatterplot, marker="+")
g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
g.add_legend(frameon=True)
g.legend.set_bbox_to_anchor((.61, .6))
plt.show() 

# Opinionated defaults and flexible customization

sns.relplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g"
)
plt.show() 

# 

sns.set_theme(style="ticks", font_scale=1.25)
g = sns.relplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g",
    palette="crest", marker="x", s=100,
)
g.set_axis_labels("Bill length (mm)", "Bill depth (mm)", labelpad=10)
g.legend.set_title("Body mass (g)")
g.figure.set_size_inches(6.5, 4.5)
g.ax.margins(.15)
g.despine(trim=True)

plt.show() 

#############################################
#  Overview of seaborn plotting functions   #
#############################################

# Similar functions for similar tasks

penguins = sns.load_dataset("penguins")
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
plt.show() 

sns.kdeplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
plt.show()

## 

sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
plt.show()

sns.displot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack", kind="kde")
plt.show()

sns.displot(data=penguins, x="flipper_length_mm", hue="species", col="species")
plt.show()

# Axes-level functions make self-contained plots

f, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=dict(width_ratios=[4, 3]))
sns.scatterplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species", ax=axs[0])
sns.histplot(data=penguins, x="species", hue="species", shrink=.8, alpha=.8, legend=False, ax=axs[1])
f.tight_layout()
plt.show()

# Figure-level functions own their figure

tips = sns.load_dataset("tips")
g = sns.relplot(data=tips, x="total_bill", y="tip")
g.ax.axline(xy1=(10, 2), slope=.2, color="b", dashes=(5, 2))
plt.show()

# Customizing plots from a figure-level function

g = sns.relplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", col="sex")
g.set_axis_labels("Flipper length (mm)", "Bill length (mm)")
plt.show()

# Specifying figure sizes

f, ax = plt.subplots()
plt.show()

f, ax = plt.subplots(1, 2, sharey=True)
plt.show()

g = sns.FacetGrid(penguins)
plt.show()

# Combining multiple views on the data

sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")
plt.show()

sns.pairplot(data=penguins, hue="species")
plt.show()

sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species", kind="hist")
plt.show()

###########################################
# Data structures accepted by seaborn    # 
#############################################

# https://seaborn.pydata.org/tutorial/data_structure.html