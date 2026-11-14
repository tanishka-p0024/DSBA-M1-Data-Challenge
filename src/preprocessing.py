# 1. Converting obj columns into date-time objects

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