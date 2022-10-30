#######################################
#          Bokeh    fundamentals      # 
#######################################

# https://bokeh.org/

import numpy as np # we will use this later, so import it now

from bokeh.io import output_notebook
from bokeh.plotting import figure

from bokeh.io import output_notebook, show


#Import the required packages
 from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
#Create two data arrays
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


# Create plot
plot = figure(width=400, height=400, title="Simple line plot", x_axis_label="x-axis", y_axis_label = 'y-axis')
plot.line(x,y, line_width=2, color='green')



#Show plot
output_file("line_plot1.html")
show(plot)

#Import the
from bokeh.plotting import figure, show

animals = ['lion', 'leopard', 'elephant', 'rhino', 'buffalo']
weight_tonnes = [190, 90, 3000, 2300, 590]

p = figure(x_range=animals, height=350, title="Big Five weight", x_axis_label = "Animal", y_axis_label = "Weight",
           toolbar_location=None, tools="")

p.vbar(x=animals, top=weight_tonnes, width=0.9)

p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)


# Import required packages
from bokeh.io import output_file, show
from bokeh.plotting import figure
# Create the regions to chart
x_region = [[1,1,2], [2,3,3], [2,3,5,4]]
y_region = [[2,5,6], [3,6,7], [2,4,7,8]]
# Create plot
plot = figure()
plot.patches(x_region, y_region, fill_color = ['blue', 'yellow', 'green'], line_color = 'black')
show(plot)

# Import required packages
from bokeh.io import output_file, show
from bokeh.plotting import figure
# Create x and y data points
x = [1,2,3,4,5]
y = [5,7,2,2,4]
# Create plot
plot = figure(title = "Scatter plot", x_axis_label = "Label name of x axis", y_axis_label ="Label name of y axis")
plot.circle(x,y, size = 30, alpha = 0.5)
# Add labels

# Output the plot
show(plot)

#########################################################
#  Interactive Data Visualization in Python With Bokeh  #
#########################################################

"""Bokeh Visualization Template

This template is a general outline for turning your data into a 
visualization using Bokeh.
"""
# Data handling
import pandas as pd
import numpy as np

# Bokeh libraries
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel

# Generating Your First Figure

# Bokeh Libraries
from bokeh.io import output_file
from bokeh.plotting import figure, show

# The figure will be rendered in a static HTML file called output_file_test.html
output_file('output_file_test.html', 
            title='Empty Bokeh Figure')

# Set up a generic figure() object
fig = figure()

# See what it looks like
show(fig)

######

from bokeh.io import output_file
from bokeh.plotting import figure, show

# My x-y coordinate data
x = [1, 2, 1]
y = [1, 1, 2]

# Output the visualization directly in the notebook
output_file('first_glyphs.html', title='First Glyphs')

# Create a figure with no toolbar and axis ranges of [0,3]
fig = figure(title='My Coordinates',
             plot_height=300, plot_width=300,
             x_range=(0, 3), y_range=(0, 3),
             toolbar_location=None)

# Draw the coordinates as circles
fig.circle(x=x, y=y,
           color='green', size=10, alpha=0.5)

# Show plot
show(fig)


######################################

# Bokeh libraries
from bokeh.io import output_notebook
from bokeh.plotting import figure, show

# My word count data
day_num = np.linspace(1, 10, 10)
daily_words = [450, 628, 488, 210, 287, 791, 508, 639, 397, 943]
cumulative_words = np.cumsum(daily_words)

# Create a figure with a datetime type x-axis
fig = figure(title='My Tutorial Progress',
             plot_height=400, plot_width=700,
             x_axis_label='Day Number', y_axis_label='Words Written',
             x_minor_ticks=2, y_range=(0, 6000),
             toolbar_location=None)

# The daily words will be represented as vertical bars (columns)
fig.vbar(x=day_num, bottom=0, top=daily_words, 
         color='blue', width=0.75, 
         legend='Daily')

# The cumulative sum will be a trend line
fig.line(x=day_num, y=cumulative_words, 
         color='gray', line_width=1,
         legend='Cumulative')

# Put the legend in the upper left corner
fig.legend.location = 'top_left'

# Let's check it out
show(fig)

########################

# Bokeh libraries
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import ColumnDataSource

# Output to file
output_file('west-top-2-standings-race.html', 
            title='Western Conference Top 2 Teams Wins Race')

# Isolate the data for the Rockets and Warriors
rockets_data = west_top_2[west_top_2['teamAbbr'] == 'HOU']
warriors_data = west_top_2[west_top_2['teamAbbr'] == 'GS']

# Create a ColumnDataSource object for each team
rockets_cds = ColumnDataSource(rockets_data)
warriors_cds = ColumnDataSource(warriors_data)

# Create and configure the figure
fig = figure(x_axis_type='datetime',
             plot_height=300, plot_width=600,
             title='Western Conference Top 2 Teams Wins Race, 2017-18',
             x_axis_label='Date', y_axis_label='Wins',
             toolbar_location=None)

# Render the race as step lines
fig.step('stDate', 'gameWon', 
         color='#CE1141', legend='Rockets', 
         source=rockets_cds)
fig.step('stDate', 'gameWon', 
         color='#006BB6', legend='Warriors', 
         source=warriors_cds)

# Move the legend to the upper left corner
fig.legend.location = 'top_left'

# Show the plot
show(fig)

#########################################

from bokeh.plotting import figure, output_notebook, show

# create figure
p = figure(plot_width = 400, plot_height = 400)
  
# add a circle renderer with
# size, color and alpha
p.circle([1, 2, 3, 4, 5], [4, 7, 1, 6, 3], 
         size = 10, color = "navy", alpha = 0.5)
  
# show the results
show(p) 

##

# import modules
from bokeh.plotting import figure, output_notebook, show
  
  
# create figure
p = figure(plot_width = 400, plot_height = 400)
   
# add a line renderer
p.line([1, 2, 3, 4, 5], [3, 1, 2, 6, 5], 
        line_width = 2, color = "green")
  
# show the results
show(p)

## 


# import necessary modules
import pandas as pd
from bokeh.charts import Bar, show
  
  
# read data in dataframe
df = pd.read_csv(r"D:/kaggle/mcdonald/menu.csv")
  
# create bar
p = Bar(df, "Category", values = "Calories",
        title = "Total Calories by Category", 
                        legend = "top_right")
  
# show the results
show(p)

## 


# import necessary modules
from bokeh.charts import BoxPlot, show
import pandas as pd
  
# output to notebook
output_notebook()
  
# read data in dataframe
df = pd.read_csv(r"D:/kaggle / mcdonald / menu.csv")
  
# create bar
p = BoxPlot(df, values = "Protein", label = "Category", 
            color = "yellow", title = "Protein Summary (grouped by category)",
             legend = "top_right")
  
# show the results
show(p)

## Histogram

# import necessary modules
from bokeh.charts import Histogram, show
import pandas as pd
  
# output to notebook
output_notebook()
  
# read data in dataframe
df = pd.read_csv(r"D:/kaggle / mcdonald / menu.csv")
  
# create histogram
p = Histogram(df, values = "Total Fat",
               title = "Total Fat Distribution", 
               color = "navy")
  
# show the results
show(p) 

# Scatter plot

# import necessary modules
from bokeh.charts import Scatter, show
import pandas as pd
  
# output to notebook
output_notebook()
  
# read data in dataframe
df = pd.read_csv(r"D:/kaggle / mcdonald / menu.csv")
  
# create scatter plot
p = Scatter(df, x = "Carbohydrates", y = "Saturated Fat",
            title = "Saturated Fat vs Carbohydrates",
            xlabel = "Carbohydrates", ylabel = "Saturated Fat",
            color = "orange")
   
# show the results
show(p) 