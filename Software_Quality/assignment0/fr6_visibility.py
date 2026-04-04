def fr6_visibility(sat_lat, sat_long, grn_lat, grn_long, grn_n, max_range):
    if not (len(grn_lat) == len(grn_long) == grn_n):
        return []

    all_lats = grn_lat + [sat_lat]
    all_longs = grn_long + [sat_long]

    if any(lat <= -90 or lat >= 90 for lat in all_lats):
        return []
    if any(lon <= -180 or lon >= 180 for lon in all_longs):
        return []
    

    

print(fr6_visibility(40, -30, [41, 52, 38.5], [-4, 0, -32.5], 3, 15))

