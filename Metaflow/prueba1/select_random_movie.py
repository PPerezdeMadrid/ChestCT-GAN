def select_random_movie(df):
    """Selecciona una película aleatoria según el género."""
    df['genres_list'] = df['genres'].apply(lambda x: x.split('|'))
    genre_to_filter = "Action"  # Cambia este valor si quieres filtrar otro género
    filtered_movies = df[df['genres_list'].apply(lambda genres: genre_to_filter in genres)]

    if not filtered_movies.empty:
        selected_movie = filtered_movies.sample(1).iloc[0]
        print(f"Película seleccionada: {selected_movie['movie_title']} ({selected_movie['title_year']})")
    else:
        print(f"No se encontraron películas del género: {genre_to_filter}")
