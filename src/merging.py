
import numpy as np
import pandas as pd


def add_music_concert_features(data, accounts, tracks):
    """
    Add music-account, user-level, artist-level and
    concert-genre features to the competition dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Original competition train or test dataframe.
    accounts : pandas.DataFrame
        One row per music app account.
    tracks : pandas.DataFrame
        Track playback events.

    Returns
    -------
    pandas.DataFrame
        Enriched competition dataframe.
    """

    # Work on copies to avoid modifying original data
    data = data.copy()
    accounts = accounts.copy()
    tracks = tracks.copy()


    data = data.merge(
        accounts,
        on="music_app_user_id",
        how="left",
        validate="many_to_one"
    )

    # -----------------------------------------
    # 2. Track completion ratio
    # -----------------------------------------

    tracks["completion_ratio"] = (
        tracks["played_ms"] / tracks["track_ms"]
    )

    # -----------------------------------------
    # 3. Overall user-level features
    # -----------------------------------------

    # user_music_features = (
    #     tracks.groupby("music_app_user_id")
    #     .agg(
    #         unique_artists=("artist_id", "nunique"),
    #         unique_genres=("genre", "nunique"),
    #         saved_tracks=("saved", "sum"),
    #         saved_rate=("saved", "mean"),
    #         autoplay_count=(
    #             "source",
    #             lambda x: (x == "autoplay").sum()
    #         ),
    #         search_count=(
    #             "source",
    #             lambda x: (x == "search").sum()
    #         )
    #     )
    #     .reset_index()
    # )

    # data = data.merge(
    #     user_music_features,
    #     on="music_app_user_id",
    #     how="left",
    #     validate="many_to_one"
    # )

 

    user_artist_features = (
        tracks.groupby(["music_app_user_id", "artist_id"])
        .agg(
            artist_play_count=("track_id", "count"),
            artist_saved_count=("saved", "sum"),
            artist_saved_rate=("saved", "mean"),
            avg_completion_rate=("completion_ratio", "mean")
        )
        .reset_index()
    )

    data = data.merge(
        user_artist_features,
        on=["music_app_user_id", "artist_id"],
        how="left",
        validate="many_to_one"
    )

 

    user_genre_features = (
        tracks.groupby(["music_app_user_id", "genre"])
        .agg(
            genre_play_count=("track_id", "count"),
            genre_saved_count=("saved", "sum"),
            genre_saved_rate=("saved", "mean"),
            genre_avg_completion=("completion_ratio", "mean")
        )
        .reset_index()
    )



    concert_genres = (
        data[["concert_id", "artist_id", "genres"]]
        .drop_duplicates()
        .assign(
            genre=lambda x: x["genres"].str.split("|", regex=False)
        )
        .explode("genre")
        [["concert_id", "artist_id", "genre"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

  

    all_user_concert_genres = (
        accounts[["music_app_user_id"]]
        .drop_duplicates()
        .merge(concert_genres, how="cross")
    )



    user_concert_genres = all_user_concert_genres.merge(
        user_genre_features,
        on=["music_app_user_id", "genre"],
        how="left",
        validate="many_to_one"
    )

    # No listening events for a genre means zero plays/saves
    user_concert_genres[
        ["genre_play_count", "genre_saved_count"]
    ] = user_concert_genres[
        ["genre_play_count", "genre_saved_count"]
    ].fillna(0)


    user_concert_features = (
        user_concert_genres
        .groupby(
            ["music_app_user_id", "concert_id", "artist_id"]
        )
        .agg(
            concert_genre_total_plays=("genre_play_count", "sum"),
            concert_genre_total_saved=("genre_saved_count", "sum"),
            avg_genre_plays=("genre_play_count", "mean"),
            max_genre_plays=("genre_play_count", "max")
        )
        .reset_index()
    )

 
    data = data.merge(
        user_concert_features,
        on=["music_app_user_id", "concert_id", "artist_id"],
        how="left",
        validate="many_to_one"
    )

    # -----------------------------------------
    # 11. Return enriched dataframe
    # -----------------------------------------

    return data

