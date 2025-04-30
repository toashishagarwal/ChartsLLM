# About ChartsLLM
Dynamically generate appropriate charts based on a given CSV data & Plotly. Though the project demonstrates the csv as input type, it could very well be in any format or in-memory data.
It contains 2 implementations - 
* Rules-based approach
* LLM-based approach

# 1. Rules-based approach
In this approach, the csv data is scanned to identify categorical, numerical & datetime columns. Further it tries to check for hints of geocoded, binary & Id columns. 
It then goes through a complex rules for these columns to suggest different chart types that would be appropriate for given data. 
When the project is run, after the csv data is scanned -
- user is provided a menu of appropriate chart types alongwith their X,Y axes
- user selects a particular item
- the corresponding chart is plotted in a new browser tab

### Setup Instructions
* Create python virtual env using dependencies in requirements.txt

### Run Instructions
* python2 main.py

# 2. LLM-based approach
In this approach, OpenAI's GPT-4 model is used alongwith a prompt template to read the samples of actual csv data. Then LLM makes a suggestion for appropriate chart type.
Once the sample csv is uploaded, the project -
- gives a preview of the file
- suggest appropriate chart type for uploaded csv data
- chooses appropriat defaults for X, Y axes
- gives an option to user to change the X & Y axis
- grouping of data

### Technologies used
- Langchain
- Python
- OpenAI LLM

### Setup Instructions
* Create python virtual env using dependencies in requirements.txt

### Run Instructions
* streamlit run chartapp.py
