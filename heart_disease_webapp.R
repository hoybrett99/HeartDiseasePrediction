# Webapp for detecting heart disease

# Import libraries
library(shiny)
library(data.table)
library(randomForest)

# Setting working directory
setwd("C:\\Users\\hoybr\\Documents\\data_projects\\heart disease\\updated\\2022")

# Read in the RF model
model <- readRDS("logistic_model.rds")

# Training set
TrainSet <- read.csv("training.csv", header = TRUE)
TrainSet <- TrainSet[,-1] %>% 
  select(Sex, HadArthritis, HadStroke, HadKidneyDisease, HadDiabetes,
         ChestScan, AgeCategory)
TrainSet %>% head()
unique(TrainSet$HadDiabetes)

####################################
# User interface                   #
####################################

# User interface
# User interface
ui <- pageWithSidebar(
  
  # Page header
  headerPanel('Heart Disease Detector'),
  
  # Input values
  sidebarPanel(
    HTML("<h3>Input parameters</h3>"),
    selectInput("Sex", 
                label = "Sex",
                choices = list("Female" = 1, "Male" = 2),
                selected = 1
    ),
    selectInput("HadStroke", 
                label = "Stroke", 
                choices = list("Yes" = 1, "No" = 0),
                selected = 0),
    selectInput("HadArthritis", 
                label = "Arthritis",
                choices = list("Yes" = 1, "No" = 0),
                selected = 0),
    selectInput("HadKidneyDisease", 
                label = "Kidney Disease",
                choices = list("Yes" = 1, "No" = 0),
                selected = 0),
    selectInput("HadDiabetes", 
                label = "Diabetes",
                choices = list("No" = 0, 
                               "Yes" = 1, 
                               "Yes, but only during pregnancy (female)" = 2,
                               "No, pre-diabetes or borderline diabetes" = 3),
                selected = 0),
    selectInput("ChestScan", 
                label = "Chest Scan",
                choices = list("Yes" = 1, "No" = 0),
                selected = 1),
    numericInput("AgeCategory", 
                 label = "Age", 
                 value = 24,
                 min = 18,
                 max = 100),
    actionButton("submitbutton", "Submit", class = "btn btn-primary")
  ),
  
  mainPanel(
    tags$label(h3('Status/Output')), # Status/Output Text Box
    verbatimTextOutput('contents'),
    tableOutput('tabledata') # Prediction results table
  )
)

####################################
# Server                           #
####################################

# Read in the logistic regression model
model <- readRDS("logistic_model.rds")

# Function to convert age to category
age_to_category <- function(age) {
  if (age >= 18 && age <= 24) return("Age18to24")
  else if (age >= 25 && age <= 29) return("Age25to29")
  else if (age >= 30 && age <= 34) return("Age30to34")
  else if (age >= 35 && age <= 39) return("Age35to39")
  else if (age >= 40 && age <= 44) return("Age40to44")
  else if (age >= 45 && age <= 49) return("Age45to49")
  else if (age >= 50 && age <= 54) return("Age50to54")
  else if (age >= 55 && age <= 59) return("Age55to59")
  else if (age >= 60 && age <= 64) return("Age60to64")
  else if (age >= 65 && age <= 69) return("Age65to69")
  else if (age >= 70 && age <= 74) return("Age70to74")
  else if (age >= 75 && age <= 79) return("Age75to79")
  else if (age >= 80) return("Ageplus80")
  else return(NA)
}

server <- function(input, output, session) {
  
  # Input Data
  datasetInput <- reactive({  
    req(input$submitbutton > 0)
    
    # Create a data frame from inputs
    test <- data.frame(
      Sex = as.numeric(input$Sex) - 1,  # 0 for Female, 1 for Male
      HadStroke = as.numeric(input$HadStroke),
      HadArthritis = as.numeric(input$HadArthritis),
      HadKidneyDisease = as.numeric(input$HadKidneyDisease),
      Diabetes = as.numeric(input$HadDiabetes),
      ChestScan = as.numeric(input$ChestScan),
      GeneralHealthPoor = 0  # Adding dummy variable
    )
    
    # Add age category columns
    age_cat <- age_to_category(input$AgeCategory)
    age_categories <- c("Age18to24", "Age25to29", "Age30to34", "Age35to39", "Age40to44", "Age45to49", 
                        "Age50to54", "Age55to59", "Age60to64", "Age65to69", "Age70to74", "Age75to79", "Ageplus80")
    for (cat in age_categories) {
      test[[cat]] <- as.numeric(age_cat == cat)
    }
    
    # Make predictions
    pred_prob <- predict(model, test, type = "response")
    pred_class <- ifelse(pred_prob > 0.5, "Yes", "No")
    
    # Create output data frame
    Output <- data.frame(
      Prediction = pred_class,
      Probability = round(pred_prob, 2)
    )
    
    Output
  })
  
  # Status/Output Text Box
  output$contents <- renderPrint({
    if (input$submitbutton > 0) { 
      "Calculation complete." 
    } else {
      "Server is ready for calculation."
    }
  })
  
  # Prediction results table
  output$tabledata <- renderTable({
    req(input$submitbutton > 0)
    isolate(datasetInput())
  })
  
}


####################################
# Create the shiny app             #
####################################
shinyApp(ui = ui, server = server)
