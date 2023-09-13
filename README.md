# Shiny Website Application

# Author: Ella Kim

# Email: [kimella\@grinnell.edu](mailto:kimella@grinnell.edu){.email}

# Final edit date: 2022/11/20

# Update:

This program is the Shiny app, connecting the Scriptable Bibliometrics written in Python to the D3 visualizations in JavaScript. It provides users the three options: (1) Search by keywords from an academic database (2) Load their own data in from an academic database (bibtex file import) (3) Import previously generated author and keyword json files from this app

(1) and (2) will generate the json and csv files for the data. All three options direct the users to the Data Visualization page, where users can interact with the author and keyword network visualizations, highlighting or selecting nodes of their interest, or selecting words from the drop-down box.

While not tested or fully implemented, the website should also aim to allow users to incorporate the FNI scores to their data and visualizations.

# Requirements

-   RStudio / Shiny
-   Having Scrapper API key (<https://www.scraperapi.com>) if user want to use option 1, for the Scrapper API will avoid blocking.
-   Access to an educational WiFi network subscribed to Elsevier Services (necessary for using scripts that connect to the Scopus API) if user want to add FNI information to the researchers file or choose option 3.
-   For option 2, the uploaded file need to have column names consistent with the sample files if user use the pre-defined 'scopus' or 'acm' option. Or they could self design their options.

# Setting up

-   Setting up the condition for FNI wrapper provided by fellow group.
    1.  Clone this repository: `git clone https://github.com/ellaseonho/fni`.

    2.  Create a new Shiny application on RStudio.

    3.  Make sure all the files cloned from the repository are moved to this application file. Be sure that you have the files .Rprofile and renv/activate.R as well.

    4.  Install the packages the `global.R` file has.

    5.  Configure Scopus API throught pybliometrics (refer to the [Pybliometrics documentation](https://pybliometrics.readthedocs.io/en/stable/) for more information on how to configure your API keys).
-   Setting up the condition for this wrapper.
    1.  Copy files from 'Emerging-Topics-Score/scripts' to the same folder of this wrapper file. (To make the import module works)

    2.  Download all the packages imported in the wrapper.

    3.  Type in the console `renv::restore()`. Enter Y to ensure the project is activated before restore, thereby ensuring that renv::restore() restores package into the project library as expected.

# Files

high level description chart or image to understanding connections 1. `global.R` : This file contains the different functions 2. `ui.R` 3. `server.R`

# Interactive Functions

-   In order to enable interactivity in the visualizations (or any components of the Shiny app in general that need something to happen only when something else occurs), you will need to learn reactive functions, like `observeEvent` `observe` and `eventReactive`.

# D3 and Shiny : Understanding the Connection

-   In your R project directory, add a new folder (directory) named "www". In this is where you will put your d3 script. When calling script src in `ui.R` you do not need to type the www path to source it.

-   When working on the D3 visualization, instead of working in a `.html` file (as most D3 tutorials start from), I suggest you to start sourcing scripts in your `ui.R` file in R Shiny directly instead. For a Shiny dashboard of our instance, a html file is unnecessary as they can be easily and better worked in `ui.R`. I have a sample program of basic communications between R and D3 called '1128_test' in git repository. This will help you visualize how the wrapper message handler works, and how the `d3.js` file is read.

-   For creating a svg element in `d3.js`, select the div id you declare in your `ui.R` file. In your files:

```{r ui}
tags$div(id = "viz_area")
```

```{js d3}
var svg = d3.select("#viz_area").append("svg")
```

-   Note the above two are simplified examples of declaring a svg element.

-   Make sure you src your d3 version script links into the `ui.R` as well (and all other needed scripts). (e.g. `tags$script(src = "https://d3js.org/d3.v4.js"`)

-   d3 visualizations can easily communicate with one source of data (i.e. one `.json` `.csv` etc. file individually) through the r2d3 interface. If you have one data set you are working with for the d3 visualization, I suggest you to refer to "<https://rstudio.github.io/r2d3/>" and utilize r2d3. However, our network visualizations require separate files generated from the Python Script Bibliometrics. In this case, you will need to utilize a custom message handler that wraps around your d3 script. In the `server.R`, utilize sendCustomMessage function.

-   In order to make d3 visualiations communicate with more than one dataset, you will need to use the wrapper of custommessage handler around the d3 visualization. In addition, notice in the `d3.js` that there is a line :
`Shiny.setInputValue("made_array", Array.from(keywordsArray))` 
This enables the specific array "keywordsarray" in the `d3.js` to also update a variable called "made_array" in the shiny server. On shiny side, you want to make sure that this updates by setting it as 
`input$made_array` for the choices listed in the Shiny's filtering option.

# Work in progress

-   The user interface is identified and created but the functions for calling the search in Python Script Bibliometrics is limited in implementation on the Shiny server side. This is due to (1) some functions are unable to run on our side so it must be tested from another place (more information on Zihao's readme file about this). The corpus functions are not tested from this side.

-   Visualization by hovering over multiple nodes works in connecting. But selecting nodes specifically does not update the keywords array.   

-   The renv of this work cannot directly be placed and run on another new R environment from another computer. Suggestion is to copy the ui, server, global, python script, and www directory w/ d3 visualization, all to a new R project file and run from there. If you encounter issues, please contact `kimella@grinnell.edu`.


