from metaflow import FlowSpec, step, Parameter
import pandas as pd
import random
"""
python movie_selection.py run --csv_file movies.csv

"""

class MoviesPipeline(FlowSpec):

    csv_file = Parameter("csv_file", help="Path to the movies CSV file")

    @step
    def start(self):
        from load_csv import load_csv
        self.df = load_csv(self.csv_file)
        self.next(self.add_column)

    @step
    def add_column(self):
        from add_years_column import add_years_column
        self.df = add_years_column(self.df)
        self.next(self.select_movie)

    @step
    def select_movie(self):
        from select_random_movie import select_random_movie
        select_random_movie(self.df)
        self.next(self.end)

    @step
    def end(self):
        """Fin del pipeline."""
        print("Pipeline finalizado.")

if __name__ == "__main__":
    MoviesPipeline()

