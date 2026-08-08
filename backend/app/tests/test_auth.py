def test_auth_flow(client):
    # 1. Register a new Citizen user
    reg_payload = {
        "email": "testcitizen@disaster.org",
        "password": "Password@123",
        "full_name": "Test Citizen",
        "role": "Citizen",
    }
    res_reg = client.post("/auth/register", json=reg_payload)
    assert res_reg.status_code == 201
    user_data = res_reg.json()
    assert user_data["email"] == "testcitizen@disaster.org"
    assert user_data["role"] == "Citizen"

    # 2. Login
    login_payload = {
        "username": "testcitizen@disaster.org",
        "password": "Password@123",
    }
    res_login = client.post("/auth/login", json=login_payload)
    assert res_login.status_code == 200
    token_data = res_login.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # 3. Access /auth/me
    headers = {"Authorization": f"Bearer {token}"}
    res_me = client.get("/auth/me", headers=headers)
    assert res_me.status_code == 200
    me_data = res_me.json()
    assert me_data["email"] == "testcitizen@disaster.org"


def test_responder_login_success(client):
    login_payload = {
        "username": "responder@disaster.org",
        "password": "Responder@123",
    }
    res = client.post("/auth/login", json=login_payload)
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["role"] == "Responder"
    assert data["full_name"] == "Lead Disaster Responder"


def test_responder_login_invalid_password(client):
    login_payload = {
        "username": "responder@disaster.org",
        "password": "WrongPassword123",
    }
    res = client.post("/auth/login", json=login_payload)
    assert res.status_code == 401
    assert res.json()["detail"] == "Incorrect email or password"


def test_unauthenticated_me_request_rejected(client):
    res = client.get("/auth/me")
    assert res.status_code == 401

