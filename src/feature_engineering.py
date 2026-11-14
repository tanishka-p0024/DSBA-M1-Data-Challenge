import numpy as np
import pandas as pd
import preprocessing

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


# Now making combined features :
# build intial features:
 
def build_ifeatures(data):
    data = lead_time(data)
    data = distance(data)
    data = total_price(data)
    data = city_match(data)
    data = has_discount(data)
    data = has_music_app(data)

    return data