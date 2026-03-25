def orbital_position_management(altitudes, n, min_alt, max_alt, dev_thres):
    if n == 0 or len(altitudes) != n:
        return 0.0, False, False, False
    
    avg_alt = sum(altitudes) / n
    acurrent = altitudes[0]
    
    within_operational_range = min_alt <= acurrent <= max_alt
    dev_alert = abs(acurrent - avg_alt) > dev_thres
    reboost = acurrent < min_alt
    
    return avg_alt, within_operational_range, dev_alert, reboost