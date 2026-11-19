import pytest
from io import BytesIO


def admin_headers(client):
    response = client.post('/auth/login', json={'username': 'admin@disaster.org', 'password': 'Admin@123'})
    assert response.status_code == 200
    return {'Authorization': 'Bearer ' + response.json()['access_token']}

def test_incident_attachments_lifecycle(client):
    # 1. Create a test incident
    payload = {
        "title": "Incident with attachments",
        "description": "Earthquake tremors felt near downtown",
        "latitude": 34.05,
        "longitude": -118.24,
        "address_text": "Downtown LA"
    }
    response = client.post("/incidents", data=payload)
    assert response.status_code == 201
    incident_id = response.json()["id"]

    # 2. Upload valid attachments
    file1 = (BytesIO(b"dummy pdf content"), "report.pdf")
    file2 = (BytesIO(b"dummy image content"), "photo.png")
    
    upload_res = client.post(
        f"/incidents/{incident_id}/attachments",
        headers=admin_headers(client),
        files=[
            ("files", ("report.pdf", file1[0], "application/pdf")),
            ("files", ("photo.png", file2[0], "image/png")),
        ]
    )
    assert upload_res.status_code == 200
    attachments_data = upload_res.json()
    assert len(attachments_data) == 2
    assert attachments_data[0]["file_name"] == "report.pdf"
    assert attachments_data[1]["file_name"] == "photo.png"

    # 3. Retrieve attachments list
    list_res = client.get(f"/incidents/{incident_id}/attachments")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 2

    # 4. Verify in incident detail
    detail_res = client.get(f"/incidents/{incident_id}")
    assert detail_res.status_code == 200
    assert len(detail_res.json()["attachments"]) == 2

    # 5. Verify in response brief
    brief_res = client.get(f"/incidents/{incident_id}/brief")
    assert brief_res.status_code == 200
    assert len(brief_res.json()["attachments"]) == 2


def test_invalid_attachment_types(client):
    # 1. Create a test incident
    payload = {
        "title": "Incident for invalid files",
        "description": "Flood warnings",
        "latitude": 34.05,
        "longitude": -118.24,
        "address_text": "Downtown LA"
    }
    response = client.post("/incidents", data=payload)
    incident_id = response.json()["id"]

    # 2. Upload invalid extension (.exe)
    invalid_file = (BytesIO(b"malicious payload"), "script.exe")
    upload_res = client.post(
        f"/incidents/{incident_id}/attachments",
        files=[("files", ("script.exe", invalid_file[0], "application/x-msdownload"))]
    )
    assert upload_res.status_code == 400
    assert "Unsupported file type" in upload_res.json()["detail"]


def test_oversized_attachment(client):
    # 1. Create a test incident
    payload = {
        "title": "Incident for large files",
        "description": "Wildfire",
        "latitude": 34.05,
        "longitude": -118.24,
        "address_text": "Downtown LA"
    }
    response = client.post("/incidents", data=payload)
    incident_id = response.json()["id"]

    # 2. Upload a file exceeding 10MB limit
    oversized_data = b"0" * (11 * 1024 * 1024)  # 11 MB
    large_file = (BytesIO(oversized_data), "huge_doc.pdf")
    upload_res = client.post(
        f"/incidents/{incident_id}/attachments",
        files=[("files", ("huge_doc.pdf", large_file[0], "application/pdf"))]
    )
    assert upload_res.status_code == 413
    assert "exceeds maximum allowed size" in upload_res.json()["detail"]


def test_citizen_cannot_modify_incident_or_attach_evidence(client):
    """The backend must enforce the same privileges as the frontend."""
    created = client.post("/incidents", data={
        "description": "Water levels rising near main road",
        "latitude": 28.61, "longitude": 77.20
    })
    assert created.status_code == 201
    incident_id = created.json()["id"]

    # Unauthenticated write attempts must not be allowed.
    anonymous_status = client.patch(f"/incidents/{incident_id}", json={"status": "Completed"})
    assert anonymous_status.status_code == 401
    anonymous_attachment = client.post(
        f"/incidents/{incident_id}/attachments",
        files=[("files", ("note.pdf", BytesIO(b"test"), "application/pdf"))]
    )
    assert anonymous_attachment.status_code == 401

    registered = client.post("/auth/register", json={
        "email": f"non-responder-{incident_id}@example.net",
        "password": "CitizenSecurePassword2026!",
        "full_name": "Prototype Citizen",
        "role": "Citizen",
    })
    assert registered.status_code == 201
    login = client.post("/auth/login", json={
        "username": f"non-responder-{incident_id}@example.net",
        "password": "CitizenSecurePassword2026!",
    })
    assert login.status_code == 200
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    assert client.patch(f"/incidents/{incident_id}", json={"status": "Completed"}, headers=headers).status_code == 403
    assert client.post(
        f"/incidents/{incident_id}/attachments",
        headers=headers, files=[("files", ("note.pdf", BytesIO(b"test"), "application/pdf"))]
    ).status_code == 403
    assert client.get(f"/incidents/{incident_id}").json()["status"] == "Pending"
