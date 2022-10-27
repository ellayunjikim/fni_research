
## Data                                                                    ##
#############################################################################
updateData <- rjson::fromJSON(file = "testdata/updateData.json")
newPubData <- rjson::fromJSON(file = "testdata/newPub.json")
authorData <- rjson::fromJSON(file = "testdata/comprehensive_nodes.json")


## Begin app server
#############################################################################
server <- function(input, output,session) {
  
  # --- conda env set up (DO NOT EDIT) --- # 
  # 
  # # create a new environment 
   #reticulate::conda_create("download_test_0714_conda")
  
  # install pip
  #reticulate::conda_install("download_test_0714_conda", "pandas")
  
  # # indicate that we want to use a specific condaenv
  reticulate::use_condaenv('download_test_0714_conda', required = TRUE)

  ## SERVER LOGIC 
  #################################################################
  #
  # reticulate::source_python('0728_wrapper.py')
  
  reticulate::source_python('1003_wrapper.py')
  
  #processbibtex <- reactive(process_bibtex(input$upload$datapath))
  #renderedtime <- 0
  Values <- reactiveValues(timeval = "2022_07_14_test",
                           kwdata = data.frame (Keywords  = c("kw1", "kw2", "kw3"),
                                                ID = c("id1", "id2", "id3")),
                           array_of_kws = list(),
                           array_of_authors = list()
  )
  
  
  observeEvent(input$linkhere, {
    updateTabsetPanel(session, "home", "defaultsearch")
  })

  observeEvent(input$startsearch_scopus, {
   #  returnedText = search_from_scopus(input$input_scopus_kw, "/Users/ellakim/Desktop/ShinyMAP/0906_app",apikey_file, countries_file, year, corpus_file_path)
  })
  
  #1st option: searchData by google scholar
  searchDataServer("search1", parentSession = session)
  
  #2nd option: loadDataServer("load1", parentSession = session)
  loadDataServer("load2", parentSession = session)
  
  #3rd option: load data w corpus loadDataServer("load1", parentSession = session)
  kw_json <- eventReactive(input$reload, {
    kw_json <- rjson::fromJSON(file = input$json_kwfile$datapath) #updatedata
  })
  author_json <- eventReactive(input$reload, {
    author_json <- rjson::fromJSON(file = input$json_authorfile$datapath) # comp_nodes
  })
  
  observeEvent(input$reload,{
    updateTabsetPanel(session, "pages", selected = "keywordsnetwork")
  })
  #3rd option: If user already has JSON File to upload directly nodes and edges of
  
  # D3 viz working! 
  # if user presses "Create Graph!" Button for Keyword Network, first time
  # it shows up the kw network (isChanged is false)
  observeEvent(input$createkw, {
    # if (is.null(input$dynamicfilter))
    # print(kw_json()$nodes)
    # isolate(data$author_json)
    session$sendCustomMessage(type = "testver",
                              message = list(g = kw_json(),
                                             b = newPubData,
                                             a = author_json(),
                                             u = input$dynamicfilter,
                                             c = FALSE))
  })
  
  # if user presses keywords to filter, kw network filters and 
  # nodes selected color change
  observeEvent(input$dynamicfilter, {
    # print(input$dynamicfilter)
    session$sendCustomMessage(type = "testver",
                              message = list(g = kw_json(),
                                             b = newPubData,
                                             a = author_json(),
                                             u = input$dynamicfilter,
                                             c = TRUE))
  })
  
  # array_kw <- list()
  observeEvent(input$createkw,{
   # Values$array_of_kws <-  kw_json()$nodes%>%spread_all%>%select(keywords)
     array_of_kws <-  kw_json()$nodes%>%spread_all%>%select(keywords) #sample file ver.
    updateSelectInput(session, "dynamicfilter", 
                      label = "Below are the keywords of your network: ",
                      choices = array_of_kws$keywords, # how to write  
                      selected = NULL)
  })
  
  # 1. Create reactive to the brushed nodes by user
  # 2. Update based on the made_array change event
  # bascially populate the choices corresponding to the keywords selected 
  observeEvent(input$made_array, {
    Values$kwdata <- data.frame(input$made_array)
    # print(Values$kwdata)
    # output$futureData <- renderTable({ Values$kwdata })
    updateSelectInput(session, "dynamicfilter", 
                      label = "Below are the keywords of your network: ",
                      choices = array_of_kws$keywords, 
                      selected = input$made_array)
  })
  
  observe({
    update_array <<- input$dynamicfilter
    update_author_array <<- input$authorfilter
    # print(update_array)
    # print(input$made_array)
  })
  
  # update author filtering based on 
  # 1. users' selection of nodes in author array
  # 2. users' selection of nodes in keywords array
  # to filter to authors who published of those keywords. 
  # (connected directly to data table making--- how to do this)
  #observeEvent(input$made_author_array, {
  observe({
    # Values$array_of_authors <- author_json()$nodes%>%spread_all%>%select(author)
    array_of_authors <- author_json()$nodes%>%spread_all%>%select(author) #sample file ver.
    updateSelectInput(session, "authorfilter",
                      label = "Below are the authors of your network: ",
                      choices = array_of_authors$author,
                      selected = NULL) #input$made_author_array)
  })
  

  # fromJSON(file = input$Json$datapath)
  # observeEvent (input$selected_keyword, {
  #       array_kw <- append(array_kw, input$selected_keyword)
  #       updateCheckboxGroupInput(session, "dynamicbox",
  #                                  choices = array_kw)
  #       print(input$made_array)
  # })
  
  # observeEvent(input$made_array, {
  #   result_array <- array(data = input$made_array)
  #   #print(result_array)
  #   #Send that json from the session to our javascript
  #   session$sendCustomMessage(type="authordata", author_data)
  #   })
  #   
  # This tells shiny to run our javascript file "drawAuthor.js" and send it to the UI for rendering
  # output$d31<- renderUI({
  #  HTML('<script type="text/javascript", src="drawAuthor.js">  </script>')
  # })
  
  
  # kw filtering 
  #output$dynamicUI <- renderUI({
  # observeEvent(input$seeviz, {
  #  selectInput(id = "select_kw",
  #             label = "Select keywords of interest: ",
  #            choices = kw_nodes()$keywords,
  #           multiple = TRUE)
  #actionButton("filterstart", "Being Search by Filter!")
  #})
  #  })
  
  
  #  observeEvent(input$seeviz, {
  # observeEvent( input$made_array, {
  #  output$dynamicUI <- renderUI({
  #   selectInput(id = "selectedkw",
  #              label = "Below are the keywords you selected. 
  #             Filter here: ",
  #           choices = input$made_array,
  #            multiple = TRUE)
  #textOutput("keywords_list")
  # })
  #  checkboxGroupInput("columns", "Choose columns", names(kw_nodes()))
  # })
  # })
  
  #  observe ({
  #   print(input$getKeywords)
  # })
  # observe({
  #   if (kw_json() == 0) return()
  #   updateSelectInput(session, "dynamicfilter",
  #                     label = "Below are the keywords of your network: ",
  #                     choices = kw_edges())
  # })
  
  # if (input$createkw != 0) {
  #   array_of_kws <-  kw_json_nodes()%>%spread_all%>%select(keywords)
  # }
  
  
  
  #output$testing <- renderText({
  # input$made_filter
  # })
  
  # kwdata <- data.frame(input$made_array)
  #  print(kwdata)
  # print(input$made_array)
  # })
  
  # Print which keywords user select from selectinput options
  #output$dynamictext <- renderText ({
  #   paste("You chose", input$dynamicfilter)
  #  })
  
  
  #observeEvent(input$filterstart, {
  # session$sendCustomMessage("kwfilter", 
  #                          input$select_kw)
  #})
  # 
  #   observeEvent(input$seeviz, {
  #   output$d33 <- renderD3({ 
  #     r2d3(data = kw_json(), # json file 
  #          d3_version = 4,
  #          script = "kwViz_0718.js", 
  #          dependencies = list("forceInABox.js", "d3.v7.min.js", "netClustering.min.js"), 
  #          container = "div_graph")
  #     })
  #   })
  #   
  
  # output$'user_clicked_node' <- renderText({
  # input$user_clicked_node
  # # })
  # 
  # observeEvent(input$authortab, {
  #   print("yes")
  #     output$authorfilter = renderUI({
  #       selectInput("auth", 
  #                   label = "Filter by author here: ",
  #                   choices = NULL,
  #                   selected = NULL, 
  #                   multiple = TRUE)
  #       })
  # })
  
  # 1. When filtered keywords update, 
  # filter data to the publications that utilize those keywords
  # get the info of everything else about that publication and display in table form
  
  # merged_data <- eventReactive(input$dynamicfilter, {
  #   
  #   # # Use tidyverse to slice the data based on the drop down input
  #   dfKw <- pubcsv_raw %>%
  #     filter(kw1 == input$dynamicfilter | 
  #              kw2 == input$dynamicfilter |
  #              kw3 == input$dynamicfilter | 
  #              kw4 == input$dynamicfilter | 
  #              kw5 == input$dynamicfilter) %>%
  #     select(bib.author, bib.pub_year, bib.abstract, bib.title, keywords, gsrank, pub_url, num_citations) %>%
  #     unique() %>%
  #     rename(Author = bib.author, 
  #            "Year of Publication" = bib.pub_year, 
  #            Abstract = bib.abstract,
  #            Title = bib.title,
  #            Keywords = keywords,
  #            "GS Rank" = gsrank,
  #            URL = pub_url,
  #            "Number of Citations" = num_citations)
  #   
  #   # filter by keyword equaling to the input keyword
  #   
  # }, ignoreNULL = FALSE) #eventReactive
  # 
  # 
  # output$table1 <- renderDataTable({
  #   rendered_table <- merged_data()
  #   DT::datatable(rendered_table)
  # })
  
  
  
  
}

