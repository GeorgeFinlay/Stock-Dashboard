import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
import pandas as pd
import numpy as np
import scipy.stats as sp
import yfinance as yf
import statsmodels.api as sm


"""
Dashboard Production/Automation file. 
Takes in from the commandline a Stock Symbol
Usage: python main <stocksymbol>
Check yahoo finance for valid Stock Symbols

System Caution: Data is run locally, be careful when changing stock dates 
that your RAM can manage this. If you're on a laptop, data exceeding 2 years is
not recommended.

To Do:
+ Multiple stock requests 
    - Drop Down menu to support multiple stock requests
+ Summary Stats
+ Plot formatting for appropriate names and axes (done)
+ Changing colour scheme for visibility
"""


if len(sys.argv) != 2:
    raise ValueError("Please provide a stock to analyse")

_stock = sys.argv[1]


class Dashboard:

    def __init__(self, _stock):
        self.stock_name = _stock
        self._stock = yf.Ticker(_stock).history(period="max")
        self.data = pd.DataFrame(self._stock['High'])
        self.data["date"] = self._stock.index
        self.data.reset_index(inplace=True)
        self.colours = {
            'background' : '#890128',
            'text': '#111111'
        }
        self.app = Dash(__name__)
        # For now, this will work with simply one dataframe chosen
        # by default. This will change when more functionality is added

    def get_summary_stats(self):
        """
        This function is designed to get some basic stats to be put into the Dashboard
        So foar only used in the layout function
        :return: A dictionary of values and outputs.
        """
        # Returns
        self.data['returns'] = self.data['High'].pct_change(1)
        yearly_return = self.data['High'].pct_change(365)
        most_recent_return = self.data['High'].pct_change(365).iloc[-1]
        returns = self.data['returns'].dropna()
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        median_return = np.median(returns)
        skewness = sp.skew(returns)

        #Trend
        analysis_data = self.data.copy()
        analysis_data_x = sm.add_constant(analysis_data.index)
        analysis_data_y = np.log(analysis_data['High'])
        model = sm.OLS(analysis_data_y, analysis_data_x, missing="drop").fit()
        trend = model.params[1]

        new_data = pd.DataFrame.from_dict({"Mean Return" : [mean_return],
                "Volatility": [volatility],
                "Median Return": [median_return],
                "Skewness": [skewness],
                "Log Trend": [trend],
                "Mean Yearly Return": [np.mean(yearly_return)],
                "Most Recent Yearly Return": [most_recent_return]})

        return [new_data, self.data]

    def make_plot(self):
        """
        Function sets up plot to be used in set_layout()
        :return: a plotly figure
        """
        fig = make_subplots(
            rows= 2, cols = 2,
            shared_xaxes=True,
            specs=[[{"type": "table"},  {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            subplot_titles= [f'Log of High data for {self.stock_name}', f'High Data for {self.stock_name}',
                             'Returns data', 'Table of stats'][::-1])

        summary = self.get_summary_stats()[0]
        fig.add_trace(
            go.Table(
                header= dict(values = summary.columns, font = dict(size = 10, color = 'black'),
                             align = "left", line_color='darkslategray'),
                cells=dict(
                    values = [summary[k].tolist() for k in summary.columns],
                    align = "left",
                    line_color='darkslategray',
                    fill_color='white'
                ),
            ),
        row = 1, col = 1
        )

        fig.add_trace(
            go.Scatter(
                x = self.data['date'],
                y = self.get_summary_stats()[1]['returns'],
                mode="lines",
                ),
            row = 1, col = 2
        )

        fig.add_trace(
            go.Scatter(
                x = self.data['date'],
                y = self.data['High'],
                mode="lines",
                ),
            row = 2, col = 1
        )

        fig.add_trace(
            go.Scatter(
                x = self.data['date'],
                y = np.log(self.data['High']),
                mode="lines",
                ),
            row = 2, col = 2
        )

        fig.update_layout(
            plot_bgcolor = self.colours['background'],
            paper_bgcolor = self.colours['background'],
            font_color = self.colours['text']
        )
        return fig

    def set_layout(self):

        self.app.layout = (
            html.Div(
                style={'backgroundColor': self.colours['background']},
                    children=[
                         html.H1(
                            children = f"{self.stock_name} Dashboard",
                            style ={
                              'textAlign': 'center',
                                'color': self.colours['text']
                             }
                            ),


                    html.Div(children= f"Data Source: Yahoo Finance",
                                style={
                                    'textAlign': 'center',
                                    'color': self.colours['text']}),



                    dcc.Graph(
                        id='graph',
                        figure=self.make_plot()
                    )
                ]))

    def run(self):
        self.make_plot()
        self.set_layout()
        self.app.run_server(debug=True)


def main():
    Dashboard(_stock).run()


if __name__ == "__main__":
    main()