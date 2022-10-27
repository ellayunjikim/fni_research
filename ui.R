
# Define UI
ui <- fluidPage(
  #Navbar structure for UI
  navbarPage("FNI APP", id = "pages", theme = shinytheme("cosmo"),
             tabPanel ("Home",id = "home", icon = icon("glyphicon glyphicon-home", lib = "glyphicon", verify_fa = FALSE),
                       fluidPage(
                         fluidRow(
                           column(12,
                                   div(h2("Expert-Centered Network Dashboard:
                                   Supporting multidisciplinary research"),
                                      style = "text-align:center; font-size:35px;"),
                                  div(h4("Ella Kim, Zihao Yu, Muqi Guo"),
                                      style="text-align:center; font-size:40px;"),
                                  div(h4(" MAP Mentor: Professor Priscilla Jimenez"),
                                      style="text-align:center; font-size:40px;"),
                                  
                                  div(h3("Grinnell College Computer Science Department"), 
                                      style="text-align:center; font-size:20px;")
                           )
                         )
                       )
             ), # home page
             navbarMenu("Search Data", icon = icon("glyphicon glyphicon-search", lib= "glyphicon", verify_fa = FALSE),
                        # tabPanel(id = "defaultsearch", "Default Search",
                        #          searchDataUI("search1")),
                        tabPanel("Search Data",
                                 fluidRow(column(12,  align="center",
                                                 h1(id='search', "SEARCH"),
                                                 tags$head(tags$style("#search{color: black;
                                 font-size: 45px;
                                 }"
                                                 ))
                                 )),
                                 searchDataUI("search1")
                                 
                        ) # tab for search data
             ), # navbar panel for search 
             navbarMenu("Load Data", icon = icon("glyphicon glyphicon-upload", lib = "glyphicon", verify_fa = FALSE),
                        tabPanel("Load Data",
                                 fluidRow(column(12,  align="center",
                                                 h1(id='load', "LOAD"),
                                                 tags$head(tags$style("#load{color: black;
                                 font-size: 45px; 
                                 }"
                                                 ))
                                 )), 
                                 loadDataUI("load2")
                        )
             ), #navbar panel for Load Data
             navbarMenu("Insert JSON", icon = icon("glyphicon glyphicon-upload", lib = "glyphicon", verify_fa = FALSE),
                        tabPanel("Insert JSON Files",
                                 fluidRow(column(12,  align="center",
                                                 h1(id='json', "Insert JSON Files"),
                                                 tags$head(tags$style("#load{color: black;
                                 font-size: 45px; 
                                 }"
                                                 ))
                                 )), 
                                 fluidRow(
                                   column(12, align = 'center',
                                          h5("If you already have previously generated JSON files to rerender visualizations, insert the two json files, 
                                                   keywords and authors respectively, below."),
                                          br(),
                                          fileInput(inputId = "json_authorfile",
                                                    label = NULL,
                                                    buttonLabel = "Upload Author JSON...",
                                                    multiple = FALSE,
                                                    width = '400px',
                                                    accept = c(".json")),
                                          fileInput(inputId = "json_kwfile",
                                                    label = NULL,
                                                    buttonLabel = "Upload Keywords JSON...",
                                                    multiple = FALSE,
                                                    width = '400px',
                                                    accept = c(".json"))
                                       )
                                 ),
                                 fluidRow(column(12,  align="center",
                                                 actionButton("reload", "RELOAD VISUALIZATION")))
                                )
                                                
                        ), #insert json
             navbarMenu("Visualizations", icon = icon("glyphicon glyphicon-globe", lib = 'glyphicon', verify_fa = FALSE),
                        #tags$style(button_color_css),
                        # tags$head(tags$script(src = "0726_dataviz.js")),
                        
                        tabPanel(value = 'keywordsnetwork', "Keywords Network",
                                 column(12, align="center",
                                        h1(id = 'visualizations', "KEYWORD VISUALIZATION"),
                                        tags$head(tags$style("#visualizations{color: black;
                                                         font-size: 45px;
                                                         }"
                                        ))
                                 ),
                                 column(12, align = "center",
                                        actionButton("createkw", "Begin Visualization!")),
                                 sidebarLayout(
                                   sidebarPanel(
                                     fluidRow(
                                       column(12,
                                              tags$link(rel = "stylesheet", type = "text/css", href = "script.css"),
                                              h4(id = 'insertkw', "Filter keywords here : "),
                                              selectInput("dynamicfilter",
                                                          label = NULL,
                                                          choices = NULL, # should be updated on startload based on users data loaded (keywords from the bibtex file)
                                                          selected = NULL,
                                                          multiple = TRUE) #, selectinput
                                              # uiOutput(ns("authorfilter"))
                                       ))
                                   ), #sidebar panel
                                   mainPanel(
                                     fluidRow(
                                       column(12,
                                              # load D3JS library
                                              # tags$script(src ="https://d3js.org/d3.v7.min.js"),
                                              #tags$script(type = "text/javascript", src = "https://unpkg.com/force-in-a-box/dist/forceInABox.js"),                           
                                              #tags$script(src = "https://unpkg.com/netclustering/dist/netClustering.min.js"), 
                                              # tags$script(src = "https://cdn.jsdelivr.net/npm/d3-fetch@3"),
                                              tags$head(tags$script(src = "0930_dataviz.js")),
                                              # tags$head(tags$script(src = "0728_dataviz.js")),
                                              # tags$script(src = "viz0725.js"),
                                              tags$hr(), 
                                              includeHTML("viz.html"),
                                              title = "Keywords",
                                              h3("Keywords Visualization"),
                                              tags$div(id="chart"),
                                              tags$div(id="zoomView"),
                                              h4("Table of Keywords"),
                                              column (10, tableOutput("table1"))
                                       )
                                     )
                                   ) #main panel
                                 )),
                        tabPanel("Authors Network",
                                 column(12, align="center",
                                        h1(id = 'visualizations', "AUTHOR VISUALIZATION"),
                                        tags$head(tags$style("#visualizations{color: black;
                                                         font-size: 45px;
                                                         }"
                                        ))
                                 ),
                                 sidebarLayout(
                                   sidebarPanel(
                                     fluidRow(
                                       column(12,
                                              tags$link(rel = "stylesheet", type = "text/css", href = "script.css"),
                                              selectInput("authorfilter",
                                                          label = "Filter by authors here: ",
                                                          choices = NULL, 
                                                          selected = NULL,
                                                          multiple = TRUE) #, selectinput
                                              # uiOutput(ns("authorfilter"))
                                       ))
                                   ),
                                   mainPanel(
                                     tags$hr(),
                                     includeHTML("viz.html"),
                                     title = "Authors",
                                     h3("Authors Visualization"),
                                     tags$div(id="author"),
                                     column (10, tableOutput("table2"))
                                   )
                                 )
                        ) # authors tab
             ) # nav bar menu for viz
  )
)

# fluidRow(column(12, align="center",
#                 h1(id = 'visualizations', "VISUALIZATIONS"),
#                 tags$head(tags$style("#visualizations{color: black;
#                                  font-size: 45px;
#                                  }"
#                 ))
# )), 
# fluidRow(column(12, align = "center",
#                 actionButton("createkw", "Begin Visualization!"))),
# tags$hr(),
