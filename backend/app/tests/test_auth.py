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


def test_public_registration_rejects_elevated_roles(client):
    for role in ("Admin", "Responder"):
        response = client.post(
            "/auth/register",
            json={
                "email": f"unauthorized-{role.lower()}@disaster.org",
                "password": "Password@123",
                "full_name": "Unauthorized Registration",
                "role": role,
            },
        )
        assert response.status_code == 403
        assert "Citizen" in response.json()["detail"]


def test_default_public_registration_is_citizen(client):
    response = client.post(
        "/auth/register",
        json={
            "email": "test-default-role@disaster.org",
            "password": "Password@123",
            "full_name": "Default Citizen",
        },
    )
    assert response.status_code == 201
    assert response.json()["role"] == "Citizen"


def test_admin_can_create_elevated_users(client):
    # 1. Login as seeded admin
    admin_login = client.post(
        "/auth/login",
        json={"username": "admin@disaster.org", "password": "Admin@123"},
    )
    assert admin_login.status_code == 200
    token = admin_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Admin creates a new Responder
    responder_create = client.post(
        "/admin/users",
        json={
            "email": "new-responder@disaster.org",
            "password": "Password@123",
            "full_name": "New Responder",
            "role": "Responder",
        },
        headers=headers,
    )
    assert responder_create.status_code == 201
    assert responder_create.json()["role"] == "Responder"

    # 3. New Responder can log in and has Responder role
    resp_login = client.post(
        "/auth/login",
        json={"username": "new-responder@disaster.org", "password": "Password@123"},
    )
    assert resp_login.status_code == 200
    assert resp_login.json()["role"] == "Responder"

    # 4. Admin creates a new Admin
    subadmin_create = client.post(
        "/admin/users",
        json={
            "email": "subadmin@disaster.org",
            "password": "Password@123",
            "full_name": "Secondary Admin",
            "role": "Admin",
        },
        headers=headers,
    )
    assert subadmin_create.status_code == 201
    assert subadmin_create.json()["role"] == "Admin"

    # 5. Non-admin unauthenticated cannot call /admin/users
    unauth = client.post(
        "/admin/users",
        json={
            "email": "hacker@disaster.org",
            "password": "Password@123",
            "full_name": "Hacker",
            "role": "Admin",
        },
    )
    assert unauth.status_code == 401

    # 6. Citizen cannot call /admin/users (forbidden)
    citizen_login = client.post(
        "/auth/login",
        json={"username": "testcitizen@disaster.org", "password": "Password@123"},
    )
    citizen_headers = {"Authorization": f"Bearer {citizen_login.json()['access_token']}"}
    forbidden = client.post(
        "/admin/users",
        json={
            "email": "hacker2@disaster.org",
            "password": "Password@123",
            "full_name": "Hacker 2",
            "role": "Admin",
        },
        headers=citizen_headers,
    )
    assert forbidden.status_code == 403

