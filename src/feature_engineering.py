import numpy as np
import pandas as pd
from src import preprocessing

# Feature 1 : Lead difference handling 
'''At any cost we need to run this first'''

def lead_time(data):
    data = data.copy() 
    data = preprocessing.convert_to_datetime(data)

    data["days_since_sale_opened"] = (
        data["prediction_at"] - data["sale_opened_at"]
    ).dt.total_seconds() / 86400

    data["days_until_concert"] = (
        data["concert_at"] - data["prediction_at"]
    ).dt.total_seconds() / 86400

    return data


# Feature 2 that will count the distance between home-venue
def distance(data):
    data = data.copy()
    earth_rad = 6371.0088
    lat1 = np.radians(data["home_lat"])
    lon1 = np.radians(data["home_lon"])
    lat2 = np.radians(data["venue_lat"])
    lon2 = np.radians(data["venue_lon"])

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    a = np.clip(a, 0, 1)
    distance = 2 * earth_rad * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    data['distance'] = distance
    return data


# Feature 3 : Total cost
def total_price(data):
    data = data.copy()

    cost = (
        data['ticket_price'] * (1 - data['discount_pct'] / 100) + data['booking_fee']
    )
    data['total_cost'] = cost

    return data

# Feature 4 : checks if home city is venue city or not
def city_match(data):
    data = data.copy()

    data["city_match"] = (
        data["home_city"].str.strip().str.lower()
        == data["venue_city"].str.strip().str.lower()
    ).astype(int)

    return data

# Feature 5 : if a user got a discount or not

def has_discount(data):
    data = data.copy()

    data["has_discount"] = (
        data["discount_pct"] > 0
    ).astype(int)

    return data

# Feature 6 : does user got the music app account?
def has_music_app(data):
    data = data.copy()

    data["has_music_app"] = (
        data["music_app_user_id"].notna()
    ).astype(int)

    return data

def travel_search_feature(data):
    data = data.copy()

    data["is_travel_search"] = (
        data["travel_search_city"].notna()
        & data["travel_search_arrival"].notna()
    ).astype(int)

    return data

# Now making combined features :
# build intial features:
 
def build_ifeatures(data):
    data = lead_time(data)
    data = distance(data)
    data = total_price(data)
    data = city_match(data)
    data = has_discount(data)
    data = has_music_app(data)
    data = travel_search_feature(data)

    return data



# Genre feature engineering 7 

def genre_matching(data):
    data = data.copy()
    favourite = (
        data["stated_favourite_genre"]
        .str.strip()
        .str.lower()
    )

    concert_genres = (
        data["genres"]
        .str.lower()
        .str.split("|", regex=False)
    )

    # Feature 1 - To see if their stated genre in genre of concert
    data["fav_genre_match"] = [
        int(fav in genres) if pd.notna(fav) else np.nan
        for fav, genres in zip(favourite, concert_genres)
    ]

    # Feature 2: if the fav genre is missing or not 
    data["fav_genre_missing"] = favourite.isna().astype(int)

    return data



# Feature 8 : Missing values logic //This is non statistical hence no data leakage.//Goes before spliting
def correct_music_missing_values(data):
    data = data.copy()

    count_features = [
        "artist_play_count",
        "artist_saved_count",
        "concert_genre_total_plays",
        "concert_genre_total_saved",
        "avg_genre_plays",
        "max_genre_plays"
    ]

    genre_features = [
        "concert_genre_total_plays",
        "concert_genre_total_saved",
        "avg_genre_plays",
        "max_genre_plays"
    ]

    valid_history = data["activity_status"].isin(
        ["active", "empty"]
    )

    no_history = (
        data["has_music_app"].eq(0)
        | data["activity_status"].eq("missing")
        | data["activity_status"].isna()
    )

    # Confirmed no matching activity for valid histories
    data.loc[valid_history, count_features] = (
        data.loc[valid_history, count_features].fillna(0)
    )

    # Unavailable listening history remains unknown
    data.loc[no_history, count_features] = np.nan

    return data

# Feature 9 : Music Affinity ratios

def music_affinity_features(data):
    data = data.copy()

    # This will also avoid div by 0
    total_events = data["total_track_events"].replace(0, np.nan)

    
    data["artist_play_share"] = (
        data["artist_play_count"] / total_events
    )

    
    data["concert_genre_play_share"] = (
        data["concert_genre_total_plays"] / total_events
    )

    return data
