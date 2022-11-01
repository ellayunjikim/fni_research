library(shiny)
library(DT)
library(shinycssloaders)
library(fontawesome)
library(dashboardthemes)
library(shinydashboardPlus)
library(shinydashboard)
library(shinyscreenshot)
library(jsonlite)
library(rjson)
library(r2d3)
# --- require pkgs
library(reticulate)
library(tidyr)
library(Matrix)
library(dimensionsR)
library(pubmedR)
library(dplyr)
library(shinyWidgets)
# ---- for tidy data
library(readxl)
library(tidyverse)
library(shinyFiles)
library(base)
library(stringr)
library(js)
library(shinyjs)
library(htmlwidgets)
library(shinysense)
library(ggrepel)
library(shinythemes)
library(tidyjson)
# reticulate::source_python('1003_wrapper.py')



Values <- reactiveValues(timeval = "2022_07_14_test",
                         kwdata = data.frame (Keywords  = c("kw1", "kw2", "kw3"),
                                              ID = c("id1", "id2", "id3")),
                         array_of_kws = list()
)


## ---- LOAD ---- ## 
loadDataUI <- function(id, label= "Load") {
  ns<- NS(id)
  tagList(
    # shinythemes::themeSelector(),
    # if you have corpus file function
    fluidRow(column(12, align = 'center',
                    actionLink(ns("have_corpus"), label = "I have a corpus file to load."),
                    uiOutput(ns("if_corpus"))
                    )),
    fluidRow(column(12, align = 'center',
                    h4('Select Database: '),
                    selectInput(ns("input_db"), label = NULL,
                                choices = c('ACM', 'Scopus')),
                    h4('Select Data Type: '),
                    selectInput(ns("input_type"), label = NULL,
                                choices = c('Bibtex', 'CSV')),
                    fileInput(inputId = ns("uploadbib"), 
                              label = NULL,
                              buttonLabel = "Upload Bib/csv...",
                              multiple = FALSE,
                              accept = c("text/bib",
                                         "text/comma-separated-values,text/plain",
                                         ".bib", 
                                         "text/csv", ".csv")),
                    tags$hr()
    )),
    fluidRow(
      (column(12, align = 'center',
              actionButton(ns("startload"), "SEARCH"))
      )),
    fluidRow(
      column(1, offset = 12,
             textOutput(ns("timeval")),
             tags$head(tags$style("#timeval{color: white;
                                 font-size: 5px;
                                 }"
             ))
      ))
  )
}

loadDataServer <- function(id,parentSession){
  moduleServer(
    id,
    function(input,output,session) {
      ns <- session$ns
      
      observeEvent(input$have_corpus, {
        output$if_corpus <- renderUI({
          fluidRow(column(12, align = "center",
          fileInput(inputId = ns("uploadcorpus"),
                    label = NULL,
                    buttonLabel = "Upload Corpus...",
                    multiple = FALSE,
                    accept = c(".txt")), #file input
          fileInput(inputId = ns("apikey_file"),
                    label = NULL,
                    buttonLabel = "Upload API Key file...",
                    multiple = FALSE,
                    accept = c(".txt")) #file input
          ))
        })
      })
      
      observeEvent(input$startload, {
        withProgress(message = 'PROCESSING...', value = 0, {
          incProgress(1/2)
          rawtime <- Sys.time()
          if (input$input_type == 'Bibtex') {
            file_type <- 'bibtex'
          } else if (input$input_type == 'CSV') {
            file_type <- 'csv'
          }
          if (input$input_db == 'ACM') {
            database <- 'acm'
          } else if (input$input_db == 'Scopus') {
            database <- 'scopus'
          }
            fdir <- "nodes_and_edges_results" #session$token #paste0("loaded_data_results_", Values$timeval)
            dir.create(fdir)
            if (input$have_corpus) {
              returnedText = generate_files_with_crys_score(file_type,
                                                            input$uploadbib$datapath,
                                                            database,
                                                            fdir,
                                                            input$apikey_file$datapath,
                                                            input$uploadcorpus$datapath)
            } else if (!(input$have_corpus))
            returnedText= generate_files_without_crys_score(file_type,
                                                            input$uploadbib$datapath,
                                                            database,
                                                            fdir)
          print(paste("Your file has been converted and downloaded."))
          Sys.sleep(5)})
        month = format(rawtime,"%m")
        day = format(rawtime,"%d")
        hour = format(rawtime,"%H")
        minute = format(rawtime,"%M")
        second = format(rawtime,"%S")
        if (substr(format(rawtime,"%m"), 1,1) == "0") {
          month = substr(format(rawtime,"%m"), 2,2)
        } 
        if (substr(format(rawtime,"%d"), 1,1) == "0") {
          day =  substr(format(rawtime,"%d"), 2,2)
        } 
        if (substr(format(rawtime,"%H"), 1,1) == "0") {
          hour = substr(format(rawtime,"%H"), 2,2)
        } 
        if (substr(format(rawtime,"%M"), 1,1) == "0") {
          minute = substr(format(rawtime,"%M"), 2,2)
        } 
        if (substr(format(rawtime,"%S"), 1,1) == "0") {
          second = substr(format(rawtime,"%S"), 2,2)
        }
        isolate({ Values$timeval<-
          format(rawtime, paste0("%Y_", month, "_", day, "_", hour, "_", minute, "_", second))})
      
        withProgress(message = 'RENDERING VISUALIZATION...', value = 0, {
          incProgress(5)
          updateTabsetPanel(parentSession, "pages",
                            selected = "keywordsnetwork")
        })
        
      }, ignoreNULL = TRUE, ignoreInit = FALSE)
      
      rendered = 
        output$timeval <- renderText({ paste0(Values$timeval) })
    }
  )
}

## --- CORPUS --- ##


## ---- SEARCH ---- ##

searchDataUI <- function(id, label = "Search") {
  ns<- NS(id)
  tagList(
    fluidRow(column(12, align = 'center',
                    h4('Select Database: '),
                    selectInput(ns("input_db"), label = NULL,
                                choices = c('Scopus', 'Google Scholar'),
                                selected = NULL),
                    uiOutput(ns("if_gs")),
                    uiOutput(ns("if_scopus"))
                    )
             )
    )
}

searchDataServer <- function(id,parentSession){
  moduleServer(
    id,
    function(input,output,session) {
      ns <- session$ns
      observeEvent(input$input_db, {
      if (input$input_db == 'Google Scholar') {
        removeUI({selector = "#switch"}) # remove and replace if google scholar selected
        output$if_gs <- renderUI ({
          div(id = "switch2",fluidRow(column(12, align = 'center',
                          h4('Enter your API Key (Scraper): '),
                          textInput(ns("input_api"), label = NULL, 
                                    value = NULL,
                                    placeholder = "Enter API Key here."),
                          h4('Enter your keywords : '),
                          textInput(ns("input_kw"), label = NULL, 
                                    value = NULL,
                                    placeholder = "Enter keywords here."),
                          h4('Select year range : '),
                          sliderInput(ns("range"), label = NULL,
                                      min = 2002, max = 2022, 
                                      value = c(2002,2022)),
                          tags$hr(),
                          fluidRow(column(12,  align="center",
                                          actionButton(ns("startsearch_gs"), "SEARCH")))
          )))
        })
      }
        if (input$input_db == 'Scopus') {
          removeUI({selector = "#switch2"}) 
          output$if_scopus <- renderUI ({ 
            div(id = "switch", fluidRow(column(12, align = 'center',
                            h4('Upload Scopus API Key file: '),
                            fileInput(inputId = ns("upload_scopus_api"), 
                                      label = NULL,
                                      buttonLabel = "Upload API key file...",
                                      multiple = FALSE,
                                      accept = c()),
                            h4('Enter your keywords : '),
                            textInput(ns("input_scopus_kw"), label = NULL, 
                                      value = NULL,
                                      placeholder = "Enter keywords here."),
                            h4('Select countries of interest: '),
                            textInput(ns("input_countries"), label = NULL, 
                                      value = NULL,
                                      placeholder = "Enter countries here."),
                            h4('Select year (minimum) : '),
                            textInput(ns("input_year"), label = NULL,
                                      value = NULL,
                                      placeholder = "Enter minimum year of publication..."),
                            tags$hr(),
                            fluidRow(column(12,  align="center",
                                            actionButton(ns("startsearch_scopus"), "SEARCH")))
            )))
          })
        }
      })
     
       # if user begins search for gs
      observeEvent(input$startsearch_gs, {
          # print(input$input_kw)
          # print(input$input_api)
          dir.create(session$token)
          returnedText = google_scholar_search('google_scholar', input$input_kw, input$input_api, c(input$range[1], input$range[2]), getwd()) 
          # subject to change the directory 
      })
      
      # if user begins search for scopus
      observeEvent(input$startsearch_scopus, {
        dir.create(session$token)
          returnedText = search_from_scopus(input$input_scopus_kw, 
                                            getwd(), # "~/Desktop/ShinyMAP/0906_app", # subject to change directory 
                                            apikey_file, 
                                            input$input_countries, 
                                            input$input_year, 
                                            input$uploadcorpus$datapath)
      })
      
    }) 
}

