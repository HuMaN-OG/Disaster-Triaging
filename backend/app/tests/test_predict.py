import io
from PIL import Image


def create_dummy_image_bytes():
    img = Image.new("RGB", (224, 224), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_predict_text_english(client):
    response = client.post(
        "/predict",
        data={"text_report": "Massive flood waters rising rapidly. Four people trapped on roof!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "priority_score" in data
    assert "priority_level" in data
    assert data["language_detected"] == "English"
    assert "class_probabilities" in data
    # All 6 classes must be present
    for cls in ["Flood", "Earthquake", "Urban Fire", "Wild Fire", "Landslide", "Drought"]:
        assert cls in data["class_probabilities"]


def test_predict_text_hindi(client):
    response = client.post(
        "/predict",
        data={"text_report": "बाढ़ का पानी बहुत बढ़ गया है, तुरंत मदद भेजो!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language_detected"] == "Hindi"


def test_predict_text_hinglish(client):
    response = client.post(
        "/predict",
        data={"text_report": "Bohot paani aa gaya hai ghar me, emergency rescue team bhejo jaldi!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language_detected"] == "Hinglish"


def test_predict_image_only(client):
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "disaster_type" in data
    assert data["disaster_type"] in ["Flood", "Earthquake", "Urban Fire", "Wild Fire", "Landslide", "Drought"]
    assert "confidence" in data
    assert len(data["class_probabilities"]) == 6


def test_predict_multimodal(client):
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/predict",
        data={"text_report": "Building caught urban fire, people stranded on top floors!"},
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "disaster_type" in data
    assert "priority_score" in data
    assert "priority_level" in data
    assert "confidence" in data
    assert data["language_detected"] == "English"
    assert len(data["class_probabilities"]) == 6


def test_life_safety_hinglish_variants_override(client):
    variants = [
        "fire haar jagah h aur ma faas gya hu",
        "baadh me building me fas gaya hoon",
        "ghar me pani ghusa aur bachche phas gye",
        "chhat par log atke hai madad bhejo",
        "flooding occurred and we are stranded upstairs"
    ]
    for text in variants:
        response = client.post("/predict", data={"text_report": text})
        assert response.status_code == 200
        data = response.json()
        assert data["priority_level"] == "Critical"
        assert data["priority_score"] == 95.0


def test_safe_condition_negation_non_override(client):
    response = client.post(
        "/predict",
        data={"text_report": "fire control me hai, koi ghayal nahi hai sab safe hain yahan"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["priority_level"] != "Critical"
