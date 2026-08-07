def test_dashboard_routes(client):
    # Ensure at least 1 incident exists
    client.post(
        "/incidents",
        data={
            "description": "Wild fire spreading near forest perimeter",
            "latitude": 34.0522,
            "longitude": -118.2437,
        },
    )

    # 1. Stats
    res_stats = client.get("/dashboard/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert "total_incidents" in stats
    assert "critical" in stats
    assert "pending" in stats

    # 2. Recent
    res_recent = client.get("/dashboard/recent")
    assert res_recent.status_code == 200
    recent = res_recent.json()
    assert isinstance(recent, list)
    assert len(recent) > 0

    # 3. Map
    res_map = client.get("/dashboard/map")
    assert res_map.status_code == 200
    map_data = res_map.json()
    assert isinstance(map_data, list)
    assert len(map_data) > 0
    assert "latitude" in map_data[0]
    assert "longitude" in map_data[0]

    # 4. Analytics
    res_analytics = client.get("/dashboard/analytics")
    assert res_analytics.status_code == 200
    analytics = res_analytics.json()
    assert "disaster_distribution" in analytics
    assert "priority_distribution" in analytics
    assert "daily_incidents" in analytics
    assert "average_confidence" in analytics
