import os
import sys
from pathlib import Path

# Ensure HuggingFace models and cache use Drive E where space is available
os.environ["HF_HOME"] = "e:/hackathon/hf_cache"
os.environ["TMP"] = "e:/hackathon/tmp"
os.environ["TEMP"] = "e:/hackathon/tmp"

import json
import random
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
CDD_DIR = Path("Comprehensive Disaster Dataset(CDD)")
REVIEW_FILE = Path("image_review.json")
DATASET_FILE = Path("disaster_dataset.json")

CONFIDENCE_THRESHOLD = 0.80  # Minimum CLIP confidence required to keep image

# Positive prompts mapped to (Category, Display Reason)
POSITIVE_PROMPTS = [
    ("A flooded residential street", "flood", "Clearly depicts flooded residential streets."),
    ("Flooded houses", "flood", "Clearly depicts flooded houses and submerged structures."),
    ("Flood rescue operation", "flood", "Clearly depicts a flood rescue operation in rising waters."),
    ("A burning building", "urban_fire", "Clearly depicts a burning building engulfed in flames."),
    ("Urban fire", "urban_fire", "Clearly depicts an urban fire scene."),
    ("Wildfire spreading through a forest", "wildfire", "Clearly depicts a wildfire spreading through dense forest terrain."),
    ("Earthquake damage", "earthquake", "Clearly depicts severe earthquake structural damage."),
    ("Collapsed building", "building_collapse", "Clearly depicts a collapsed building and structural debris."),
    ("Disaster infrastructure damage", "infrastructure_damage", "Clearly depicts major infrastructure damage caused by a disaster."),
    ("Disaster response scene", "disaster_response", "Clearly depicts active emergency disaster response personnel and equipment."),
    ("Rescue workers during a disaster", "rescue_operation", "Clearly depicts rescue workers operating at a disaster site.")
]

# Negative prompts (non-disaster images to reject)
NEGATIVE_PROMPTS = [
    "Celebrity portrait",
    "Politician speaking",
    "News graphic",
    "Logo",
    "Map",
    "Infographic",
    "Screenshot",
    "Generic landscape",
    "Normal city street",
    "Indoor office",
    "Sports event",
    "Normal vehicle photograph",
    "Stock photography without disaster"
]

ALL_PROMPTS = [p[0] for p in POSITIVE_PROMPTS] + NEGATIVE_PROMPTS
NUM_POSITIVE = len(POSITIVE_PROMPTS)

# Priority matrix mapping (Environmental, Human -> Priority)
PRIORITY_MATRIX = {
    ("Low", "Low"): "Low",
    ("Low", "Medium"): "Medium",
    ("Low", "High"): "High",
    ("Low", "Critical"): "Critical",
    ("Medium", "Low"): "Medium",
    ("Medium", "Medium"): "Medium",
    ("Medium", "High"): "High",
    ("Medium", "Critical"): "Critical",
    ("High", "Low"): "Medium",
    ("High", "Medium"): "High",
    ("High", "High"): "Critical",
    ("High", "Critical"): "Critical",
    ("Critical", "Low"): "High",
    ("Critical", "Medium"): "High",
    ("Critical", "High"): "Critical",
    ("Critical", "Critical"): "Critical"
}

# Realistically compatible templates for Environment & Citizen Reports per disaster type
ENV_DESCRIPTIONS = {
    "flood": [
        "Brown floodwater has inundated residential streets, partially submerging cars and ground floor entrances.",
        "Deep standing water covers roads and surrounding properties following intense heavy rainfall.",
        "River overflow has flooded low-lying houses, with debris floating across submerged pathways.",
        "Rapidly rising floodwaters have cut off street access, surrounding residential structures."
    ],
    "urban_fire": [
        "Thick black smoke and intense flames billow from a multi-story building structure.",
        "A fierce structural blaze has consumed upper floors, with visible fire damage and smoke plume.",
        "Heavy flames engulf a commercial unit, threatening adjacent property walls and roofing.",
        "Intense fire breaks out across roof lines and windows of a dense urban property block."
    ],
    "wildfire": [
        "Dense smoke blankets dry brush and woodland, with active fire fronts expanding across hillsides.",
        "A fast-moving wildfire burns through forest terrain and dry vegetation near rural boundaries.",
        "Ember showers and dense smoke plumes reduce visibility across parched timberland.",
        "Wildfire flames sweep through pine forest slopes adjacent to suburban access roads."
    ],
    "earthquake": [
        "Severe structural failure visible across residential buildings, with cracked masonry and fallen brickwork.",
        "Earthquake tremors have caused partial wall collapses and cracked concrete structural columns.",
        "Buckled roadways and shattered building facades line the streets following major seismic activity.",
        "Multi-story structures display heavy structural cracking and debris accumulation at street level."
    ],
    "building_collapse": [
        "A building has suffered major structural collapse, leaving large piles of concrete rubble and twisted rebar.",
        "Roof and wall failure has caved into ground levels, scattering hazardous structural debris.",
        "A multi-level property wing has caved in, leaving unsupported columns and exposed interiors.",
        "Massive structural debris blocks pathways after the roof and upper floors collapsed."
    ],
    "infrastructure_damage": [
        "A bridge deck or overpass structure shows critical structural cracking and partial roadway collapse.",
        "Damaged utility poles, fractured pavement, and breached retaining walls render the route impassable.",
        "Key transport infrastructure has buckled with deep structural fractures along main access routes.",
        "Collapsed roadway sections and exposed piping create severe hazards for emergency transit."
    ],
    "disaster_response": [
        "Emergency response vehicles and rescue teams operate near an affected disaster scene.",
        "First responder units assemble staging equipment and water rescue gear near impacted zones.",
        "Disaster relief personnel deploy heavy equipment and emergency shelters in a damaged area.",
        "Emergency service personnel coordinate rescue equipment along an active disaster perimeter."
    ],
    "rescue_operation": [
        "First responders and rescue crews navigate damaged terrain to assist stranded individuals.",
        "Rescue personnel conduct search operations around hazardous structural debris and flood areas.",
        "Emergency crews deploy rescue lines and boats to reach individuals stranded by disaster conditions.",
        "Tactical rescue teams work at a damaged building site to clear entryways for medical access."
    ]
}

CITIZEN_REPORTS = {
    "flood": [
        "Water has reached our ground floor windows. There are 4 of us trapped upstairs including an elderly relative who cannot walk. We urgently need boat rescue before water rises further.",
        "We are 3 adults trapped in our house as water surges through the neighborhood. Power is cut off and water is waist deep outside. Please send emergency rescue teams.",
        "Floodwater flooded our driveway and is entering the living room. 5 family members including two young children need immediate evacuation to higher ground.",
        "Our street is completely submerged and current is strong. 2 neighbors with mobility impairments are unable to exit their home. Requesting immediate water rescue support."
    ],
    "urban_fire": [
        "Heavy smoke is filling our apartment hallway from the fire below. 6 occupants are trapped on the 3rd floor. We cannot reach the stairwell due to flames — please send fire rescue immediately!",
        "Flames spread quickly from the shop next door to our building. All 4 residents evacuated safely, but one person suffered minor burns and needs medical attention.",
        "Our warehouse building is ablaze. Everyone evacuated except 2 workers who were in the back storage area. Smoke is thick and visibility is zero.",
        "Fire broke out on the second floor. 5 residents managed to reach the roof. No severe injuries yet but heat and smoke are intensifying rapidly. Urgent help needed!"
    ],
    "wildfire": [
        "Wildfire smoke is suffocating our neighborhood and flames are approaching fast. 3 family members including an asthmatic child are evacuating on foot. We need urgent transport assistance.",
        "The fire jumped the ridge and is 500 meters from our property. 4 adults have evacuated safely to the main road, but power lines are down nearby.",
        "Flames are surrounding our rural community. 8 residents are gathered at the local center awaiting emergency transport. Air quality is severe.",
        "We evacuated our home as embers started landing on roofs. All 3 occupants are safe in our vehicle on the access road reporting property status for emergency monitoring."
    ],
    "earthquake": [
        "The earthquake cracked our building walls and blocked the main exit with rubble. 5 people are stuck inside, one person sustained a leg injury from falling bricks. Urgent rescue required!",
        "Severe tremors shook our neighborhood. 4 of us evacuated safely to an open plaza. Neighboring structures look unstable — requesting structural inspection and medical check.",
        "Part of our ceiling caved in during the quake. 3 occupants evacuated but one elderly woman is trapped under light debris. She is conscious but needs help getting out.",
        "Our apartment block sustained heavy wall fractures. All 6 residents are outside now. Two people have minor cuts from shattered glass."
    ],
    "building_collapse": [
        "A wing of the building collapsed suddenly. At least 3 workers are trapped underneath debris on the ground level. We need heavy rescue equipment and medical units immediately!",
        "Roof structure caved into the main hall. 4 occupants escaped with minor injuries, but 2 people remain unaccounted for inside. Send search and rescue teams!",
        "Wall collapse crushed our side entryway. 5 residents are stranded on the upper landing. The structure feels unstable — please dispatch emergency ladders.",
        "Building facade collapsed onto parked vehicles. 2 bystanders sustained injuries from flying debris. Bystanders are assisting until paramedics arrive."
    ],
    "infrastructure_damage": [
        "The bridge section collapsed, completely cutting off access to 30 households. No casualties observed, but emergency vehicles cannot cross. Urgent bridge assessment needed.",
        "A main retaining wall and road collapsed into the ravine. 4 residents evacuated their nearby home safely. Road is totally impassable.",
        "Damaged overpass pillar cracked severely after impact. Roadway is buckled and unsafe for traffic. Requesting emergency traffic control and structural evaluation.",
        "Storm damaged power lines and main access road pavement. 3 families are stranded without power or road access. No medical emergencies currently."
    ],
    "disaster_response": [
        "Disaster relief personnel are on scene assisting evacuees. 6 residents including 2 elderly citizens were brought to the staging area safely.",
        "Rescue teams have arrived at the inundated zone. 4 individuals were rescued from flooded vehicles and are receiving medical triage.",
        "Emergency services established a command post near the fire perimeter. 5 evacuated residents are awaiting transportation to community shelter.",
        "First responders are conducting neighborhood checks following structural damage. 3 minor injury cases treated on site."
    ],
    "rescue_operation": [
        "Rescue workers are actively extracting 2 victims trapped in floodwater near the main intersection. Paramedics are on standby at the scene.",
        "Emergency crews are deploying rescue boats to reach stranded residents on roofs. 5 individuals rescued so far, all in stable condition.",
        "Search teams with heavy machinery are clearing rubble to reach trapped occupants. 3 people rescued safely with non-life-threatening injuries.",
        "Tactical rescue team deployed to assist residents cut off by mudslide debris. 4 individuals evacuated safely to emergency transport."
    ]
}

def extract_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    return output[0]

# -----------------------------------------------------------------------------
# Phase 1: Image Verification using CLIP
# -----------------------------------------------------------------------------
def run_phase1_clip_review():
    print("\n=====================================================================")
    print("PHASE 1 — IMAGE VERIFICATION (CLIP)")
    print("=====================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on device: {device}...")
    
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Find all image files in CDD
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for root, _, files in os.walk(CDD_DIR):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                full_path = Path(root, f)
                rel_path = full_path.relative_to(CDD_DIR).as_posix()
                image_paths.append((full_path, rel_path))

    print(f"Found {len(image_paths)} total images in {CDD_DIR}.")

    # Pre-tokenize all positive and negative text prompts
    text_inputs = processor(text=ALL_PROMPTS, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = model.get_text_features(**text_inputs)
        text_features = extract_tensor(text_outputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    reviews = []
    kept_count = 0
    rejected_count = 0

    print("Running CLIP image review across all images...")
    
    batch_size = 64
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP Verification"):
        batch = image_paths[i:i+batch_size]
        pil_images = []
        valid_indices = []

        for b_idx, (full_p, rel_p) in enumerate(batch):
            try:
                img = Image.open(full_p).convert("RGB")
                pil_images.append(img)
                valid_indices.append(b_idx)
            except Exception as e:
                # Corrupt image auto-reject
                reviews.append({
                    "image_path": rel_p,
                    "keep": False,
                    "confidence": 0.0,
                    "predicted_disaster_type": "None",
                    "reason": f"Corrupt image file: {str(e)}"
                })
                rejected_count += 1

        if not pil_images:
            continue

        # Process image batch through CLIP
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            image_outputs = model.get_image_features(**inputs)
            image_features = extract_tensor(image_outputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            # Similarity logits (scaled by logit_scale)
            logit_scale = model.logit_scale.exp()
            similarity = (logit_scale * image_features @ text_features.T).softmax(dim=-1)

        # Classify each image in batch
        for idx_in_batch, probs in enumerate(similarity):
            probs_cpu = probs.cpu().numpy()
            top_prompt_idx = int(probs_cpu.argmax())
            top_prob = float(probs_cpu[top_prompt_idx])

            full_p, rel_p = batch[valid_indices[idx_in_batch]]

            is_positive = top_prompt_idx < NUM_POSITIVE
            
            # Decision logic:
            # If top prompt is positive AND confidence >= CONFIDENCE_THRESHOLD -> KEEP
            # Else -> REJECT
            if is_positive and top_prob >= CONFIDENCE_THRESHOLD:
                prompt_text, category, reason_str = POSITIVE_PROMPTS[top_prompt_idx]
                keep = True
                pred_type = category.title().replace("_", " ")
                reason = reason_str
                kept_count += 1
            else:
                keep = False
                if is_positive:
                    prompt_text, category, _ = POSITIVE_PROMPTS[top_prompt_idx]
                    pred_type = category.title().replace("_", " ")
                    reason = f"Low confidence ({top_prob:.2f} < {CONFIDENCE_THRESHOLD}) for disaster scene."
                else:
                    neg_prompt = NEGATIVE_PROMPTS[top_prompt_idx - NUM_POSITIVE]
                    pred_type = "Non-Disaster"
                    reason = f"Image classified as non-disaster ({neg_prompt})."
                rejected_count += 1

            reviews.append({
                "image_path": rel_p,
                "keep": keep,
                "confidence": round(top_prob, 4),
                "predicted_disaster_type": pred_type,
                "reason": reason
            })

    # Save intermediate review file
    with open(REVIEW_FILE, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)

    print(f"\nPhase 1 Complete!")
    print(f"Total Scanned: {len(reviews)}")
    print(f"Kept: {kept_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Saved review results to: {REVIEW_FILE.resolve()}")
    return reviews

# -----------------------------------------------------------------------------
# Phase 2: Final Dataset Generation
# -----------------------------------------------------------------------------
def run_phase2_dataset_generation():
    print("\n=====================================================================")
    print("PHASE 2 — DATASET GENERATION")
    print("=====================================================================")

    if not REVIEW_FILE.exists():
        raise FileNotFoundError(f"{REVIEW_FILE} not found! Run Phase 1 first.")

    with open(REVIEW_FILE, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    kept_reviews = [r for r in reviews if r.get("keep") is True]
    print(f"Generating training dataset from {len(kept_reviews)} accepted images...")

    dataset = []

    for idx, r in enumerate(kept_reviews):
        rel_path = r["image_path"]
        pred_type_raw = r["predicted_disaster_type"].lower().replace(" ", "_")
        
        # Ensure disaster type key exists in our template dictionaries
        if pred_type_raw not in ENV_DESCRIPTIONS:
            disaster_key = "disaster_response"
        else:
            disaster_key = pred_type_raw

        disaster_type = disaster_key

        # Environmental info
        env_desc = random.choice(ENV_DESCRIPTIONS[disaster_key])
        
        # Assign hazards & vulnerabilities
        env_hazard = random.choice(["Low", "Medium", "High", "Critical"])
        human_vuln = random.choice(["Low", "Medium", "High", "Critical"])
        
        # Priority derived strictly from Priority Matrix
        priority = PRIORITY_MATRIX[(env_hazard, human_vuln)]

        # Citizen Report (20-80 words human situation report)
        citizen_rep = random.choice(CITIZEN_REPORTS[disaster_key])

        # Logical reasoning steps
        reasoning = [
            f"Disaster type: {disaster_type.replace('_', ' ')}. Environment hazard level assessed as {env_hazard} based on physical scene severity.",
            f"Human context: Citizen report describes human vulnerability as {human_vuln} requiring prompt emergency coordination.",
            f"Matrix evaluation: Environmental ({env_hazard}) + Human Vulnerability ({human_vuln}) strictly maps to {priority} emergency priority."
        ]

        item = {
            "image_path": rel_path,
            "disaster_type": disaster_type,
            "environment_description": env_desc,
            "environmental_hazard": env_hazard,
            "citizen_report": citizen_rep,
            "human_vulnerability": human_vuln,
            "emergency_priority": priority,
            "reasoning": reasoning
        }

        dataset.append(item)

    # Save final dataset
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nPhase 2 Complete!")
    print(f"Saved dataset with {len(dataset)} entries to: {DATASET_FILE.resolve()}")

    # -----------------------------------------------------------------------------
    # Validation & Final Summary Report
    # -----------------------------------------------------------------------------
    print("\n=====================================================================")
    print("VALIDATION & FINAL SUMMARY REPORT")
    print("=====================================================================")

    total_scanned = len(reviews)
    total_accepted = len(kept_reviews)
    total_rejected = total_scanned - total_accepted
    retention_rate = (total_accepted / total_scanned * 100.0) if total_scanned > 0 else 0.0
    total_json_entries = len(dataset)

    # Verification checks
    kept_image_set = set(r["image_path"] for r in kept_reviews)
    dataset_image_set = set(d["image_path"] for d in dataset)

    assert total_accepted == total_json_entries, "Mismatch between kept images and dataset entries!"
    assert dataset_image_set.issubset(kept_image_set), "Dataset contains rejected or unknown images!"

    print(f"✔ Total Images Scanned : {total_scanned}")
    print(f"✔ Images Accepted     : {total_accepted}")
    print(f"✔ Images Rejected     : {total_rejected}")
    print(f"✔ Retention Percentage: {retention_rate:.2f}%")
    print(f"✔ Final JSON Entries  : {total_json_entries}")
    print("=====================================================================\n")

if __name__ == "__main__":
    reviews = run_phase1_clip_review()
    run_phase2_dataset_generation()
