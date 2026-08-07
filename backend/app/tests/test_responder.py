def test_responder_workflow(client):
    # 1. Create a pending incident
    inc_data = {
        "description": "Landslide blocking main mountain road",
        "latitude": 30.3165,
        "longitude": 78.0322,
    }
    create_res = client.post("/incidents", data=inc_data)
    assert create_res.status_code == 201
    incident_id = create_res.json()["id"]

    # 2. Responder Login
    login_res = client.post(
        "/auth/login",
        json={"username": "responder@disaster.org", "password": "Responder@123"},
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. View Responder Pending Incidents
    pending_res = client.get("/responder/incidents", headers=headers)
    assert pending_res.status_code == 200

    # 4. Accept Incident -> Status = Assigned
    accept_res = client.post(f"/responder/{incident_id}/accept", headers=headers)
    assert accept_res.status_code == 200
    assert accept_res.json()["status"] == "Assigned"

    # 5. Update Status -> En Route
    st1_res = client.patch(
        f"/responder/{incident_id}/status",
        json={"status": "En Route", "notes": "Rescue vehicle dispatched"},
        headers=headers,
    )
    assert st1_res.status_code == 200
    assert st1_res.json()["status"] == "En Route"

    # 6. Update Status -> Rescue In Progress
    st2_res = client.patch(
        f"/responder/{incident_id}/status",
        json={"status": "Rescue In Progress", "notes": "Excavation team operating"},
        headers=headers,
    )
    assert st2_res.status_code == 200
    assert st2_res.json()["status"] == "Rescue In Progress"

    # 7. Update Status -> Completed
    st3_res = client.patch(
        f"/responder/{incident_id}/status",
        json={"status": "Completed", "notes": "Road cleared and survivors evacuated"},
        headers=headers,
    )
    assert st3_res.status_code == 200
    assert st3_res.json()["status"] == "Completed"
