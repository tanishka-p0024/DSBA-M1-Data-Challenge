# 1. Converting obj columns into date-time objects
import pandas as pd
import numpy as np
def convert_to_datetime(data):
    data = data.copy()

    date_cols = [
        "concert_at",
        "sale_opened_at",
        "prediction_at"
    ]

    for col in date_cols:
        data[col] = pd.to_datetime(data[col], utc=True)

    return data


# Only run this after applying the feature engineering step
def remove_cols(data):
    data = data.drop(['home_lat','home_lon','artist_id_x','artist_id_y',
                      'venue_lat','venue_lon','ticket_price',
                      'booking_fee','discount_pct',
                      'travel_search_city','travel_search_arrival'],axis = 1
                      )
    return data


def remove_datetime(data):
    data = data.drop(["concert_at",
    "sale_opened_at",
    "prediction_at"],axis = 1)
    return data 
