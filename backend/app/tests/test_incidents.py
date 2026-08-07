import io
from PIL import Image


def create_dummy_image_bytes():
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_incident_lifecycle(client):
    # 1. Create Incident
    img_bytes = create_dummy_image_bytes()
    data = {
        "title": "Severe Earthquake Damage",
        "description": "Building collapsed after heavy tremors.",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "address_text": "Connaught Place, New Delhi",
    }
    files = {"image": ("quake.jpg", img_bytes, "image/jpeg")}

    res_create = client.post("/incidents", data=data, files=files)
    assert res_create.status_code == 201
    inc = res_create.json()
    incident_id = inc["id"]
    assert inc["title"] == "Severe Earthquake Damage"
    assert inc["status"] == "Pending"
    assert "class_probabilities" in inc
    assert "recommendations" in inc
    assert len(inc["recommendations"]) > 0

    # 2. Get All Incidents
    res_list = client.get("/incidents")
    assert res_list.status_code == 200
    incidents = res_list.json()
    assert len(incidents) >= 1

    # 3. Get Single Incident by ID
    res_get = client.get(f"/incidents/{incident_id}")
    assert res_get.status_code == 200
    detail = res_get.json()
    assert detail["id"] == incident_id

    # 4. Admin Login & Delete Incident
    login_res = client.post(
        "/auth/login",
        json={"username": "admin@disaster.org", "password": "Admin@123"},
    )
    admin_token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {admin_token}"}

    res_del = client.delete(f"/incidents/{incident_id}", headers=headers)
    assert res_del.status_code == 204

    # 5. Confirm Deleted
    res_verify = client.get(f"/incidents/{incident_id}")
    assert res_verify.status_code == 404
