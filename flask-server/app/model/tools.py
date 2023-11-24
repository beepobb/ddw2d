import pandas as pd
from app.model.tools import MultipleLinearRegression

class Tool:
    def __init__(self):
        # Initialize the tool with default values
        # self.county = #user input
        # self.money = #user input
        self.targetCurrency
        self.df = pd.read_csv("{{ url_for('static', filename='csv/merged_dataset.csv') }}")
        self.target = "Cost of a healthy diet (PPP dollar per person per day)"
        self.featureList = ["Happiness Index", "Human Development Index", "Quality of Life Index",
                            "Purchasing Power Index", "Cost of Living Index",
                            "Property Price to Income Ratio", "Pollution Index"]

        self.model = MultipleLinearRegression()
        rundf = pd.read_csv("{{ url_for('static', filename='csv/Final Dataset Kind of .csv) }}")
        self.model.store_data(rundf, self.featureList, self.target, random_state=100, test_size=0)
        self.model.train(self.target, show=False)

    
    def findData(self, country):
        # Filter the DataFrame based on the provided country
        country_row = self.df[self.df['country'] == country]
        features_for_country = country_row[self.featureList].iloc[0]
        return features_for_country


    def predict(self, country, money):
        # Find the data for the given country
        country_data = self.findData(country)

        if country_data.empty:
            print(f"No data found for {country}.")
            return None

        # Extract features for prediction
        features_for_prediction = country_data[self.featureList]

        # Make predictions
        predicted_cost = self.model.predict(features_for_prediction, self.target)[0][0]

        
        return predicted_cost


        
        