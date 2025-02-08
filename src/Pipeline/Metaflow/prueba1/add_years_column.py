def add_years_column(df):
    """Añade la columna 'años'."""
    df['años'] = 2025 - df['title_year']
    print("Datos con la nueva columna 'años':\n", df.head())
    return df
