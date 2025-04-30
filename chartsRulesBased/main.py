import pandas as pd
from DataVizSelector import DataVizSelector

def plotGraph(csvFile):
    # Print data description
    print("*** For: ", csvFile)

    # Load your data
    df = pd.read_csv(csvFile)  # or any pandas DataFrame

    # Initialize the selector
    viz_selector = DataVizSelector(df)

    # Print data description
    print("Data Description:")
    description = viz_selector.describe_data()
    print(f"- {len(description['column_types']['categorical'])} categorical columns")
    print(f"- {len(description['column_types']['numeric'])} numeric columns")
    print(f"- {len(description['column_types']['datetime'])} datetime columns")

    # Get top visualization recommendations
    print("\nTop Recommended Visualizations:")
    recommended_viz = viz_selector.get_recommended_visualizations(top_n=5)
    for i, viz_id in enumerate(recommended_viz):
        viz_info = viz_selector.suggestions[viz_id]
        print(f"{i+1}. {viz_info['title']}: {viz_info['description']}")
    
    # Get user input for visualization selection
    while True:
        try:
            choice = int(input("\nSelect a visualization to display (enter number 1-5): "))
            if 1 <= choice <= len(recommended_viz):
                selected_viz = recommended_viz[choice-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(recommended_viz)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Generate and display the selected visualization
    viz_info = viz_selector.suggestions[selected_viz]
    print(f"\nGenerating: {viz_info['title']}")
    fig = viz_selector.generate_visualization(selected_viz)
    
    # Display the visualization
    fig.show()
    
    # Optionally save the visualization to HTML
    output_file = f"{csvFile.split('.')[0]}_{selected_viz}_visualization.html"
    fig.write_html(output_file)
    print(f"\nVisualization saved to {output_file}")

# Uncomment the file you want to analyze
#plotGraph('yourdata.csv')
plotGraph('salesdata.csv')
#plotGraph('geodata.csv')
#plotGraph('surveyresponse.csv')
#plotGraph('websitemetrics.csv')
