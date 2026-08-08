import pytest

def test_create_and_filter_incidents(client):
    # 1. Create a test incident with application_context
    payload = {
        "title": "Filter Test Incident",
        "description": "Chemical gas leak in warehouse cluster, help needed",
        "latitude": 34.05,
        "longitude": -118.24,
        "address_text": "Los Angeles Warehouse District",
        "application_context": "API Dispatch"
    }
    response = client.post("/incidents", data=payload)
    assert response.status_code == 201
    incident_data = response.json()
    incident_id = incident_data["id"]
    assert incident_data["application_context"] == "API Dispatch"

    # 2. Filter by disaster_type
    res_type = client.get("/incidents", params={"disaster_type": incident_data["disaster_type"]})
    assert res_type.status_code == 200
    assert any(inc["id"] == incident_id for inc in res_type.json())

    # 3. Filter by location search term
    res_loc = client.get("/incidents", params={"location": "Warehouse"})
    assert res_loc.status_code == 200
    assert any(inc["id"] == incident_id for inc in res_loc.json())

    # 4. Filter by application_context
    res_ctx = client.get("/incidents", params={"application_context": "API Dispatch"})
    assert res_ctx.status_code == 200
    assert any(inc["id"] == incident_id for inc in res_ctx.json())

    # 5. Fetch response brief
    res_brief = client.get(f"/incidents/{incident_id}/brief")
    assert res_brief.status_code == 200
    brief_data = res_brief.json()

    # Verify Confirmed Data
    assert brief_data["incident_id"] == incident_id
    assert brief_data["disaster_type"] == incident_data["disaster_type"]
    assert brief_data["priority_score"] == incident_data["priority_score"]
    assert brief_data["location"] == "Los Angeles Warehouse District"
    assert brief_data["latitude"] == 34.05
    assert brief_data["longitude"] == -118.24
    assert brief_data["current_status"] == "Pending"
    assert len(brief_data["timeline"]) >= 1

    # Verify AI Recommendations
    assert "risk_summary" in brief_data
    assert len(brief_data["immediate_actions"]) >= 1
    assert len(brief_data["recommended_resources"]) >= 1
