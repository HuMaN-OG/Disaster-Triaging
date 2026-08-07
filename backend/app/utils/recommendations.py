from typing import List

RECOMMENDATIONS_MAP = {
    "Flood": {
        "Critical": [
            "Deploy inflatable rescue boats and amphibious vehicles immediately",
            "Dispatch emergency medical response team for waterborne rescue",
            "Issue immediate high-ground evacuation orders",
            "Distribute life jackets and emergency flotation devices",
            "Establish clean drinking water distribution points"
        ],
        "High": [
            "Deploy rescue boat team for vulnerable populations",
            "Alert local emergency response units and medical centers",
            "Prepare temporary shelter and evacuation facilities",
            "Monitor rising water levels and flood barriers"
        ],
        "Medium": [
            "Dispatch local rescue patrol for welfare check",
            "Issue flood warning and safety advisories",
            "Pre-stage emergency relief supplies nearby"
        ],
        "Low": [
            "Monitor area via local emergency services",
            "Advise citizens to move valuables to upper levels"
        ],
        "Very Low": [
            "Standard monitoring of water drainage systems"
        ]
    },
    "Urban Fire": {
        "Critical": [
            "Dispatch multiple fire brigade engines and ladder trucks immediately",
            "Deploy advanced emergency medical services & ambulances",
            "Initiate immediate evacuation of adjacent buildings and structures",
            "Dispatch police units for perimeter isolation and traffic control",
            "Alert burn specialty centers and local hospitals"
        ],
        "High": [
            "Dispatch fire brigade units and emergency medical team",
            "Evacuate immediate building and isolate gas/power lines",
            "Set up emergency triage perimeter"
        ],
        "Medium": [
            "Dispatch local fire squad to assess and contain",
            "Evacuate affected floors and nearby units"
        ],
        "Low": [
            "Dispatch local fire department unit for inspection"
        ],
        "Very Low": [
            "Routine fire hazard inspection and report verification"
        ]
    },
    "Earthquake": {
        "Critical": [
            "Deploy specialized Search and Rescue (USAR) heavy extraction teams",
            "Dispatch mobile emergency field hospitals and medical units",
            "Deploy heavy excavation equipment for structural debris removal",
            "Conduct structural stability assessment of nearby bridges and roads",
            "Set up emergency relief camps with food, water, and shelter"
        ],
        "High": [
            "Deploy Search and Rescue team with specialized detection equipment",
            "Dispatch emergency medical response teams",
            "Inspect infrastructure integrity and clear secondary hazards"
        ],
        "Medium": [
            "Dispatch local search and rescue patrol",
            "Establish temporary triage point",
            "Check utility lines for gas leaks or electrical hazards"
        ],
        "Low": [
            "Conduct safety inspection of affected structures",
            "Advise residents on aftershock protocols"
        ],
        "Very Low": [
            "Log seismic report and verify regional building status"
        ]
    },
    "Wild Fire": {
        "Critical": [
            "Request immediate aerial water/fire retardant drops",
            "Deploy wildland firefighting task forces and bulldozer containment lines",
            "Issue mandatory evacuation orders for surrounding residential zones",
            "Establish smoke and air quality emergency health advisories"
        ],
        "High": [
            "Deploy forest firefighting units and water tankers",
            "Establish fire containment line",
            "Prepare nearby communities for potential evacuation"
        ],
        "Medium": [
            "Dispatch wildland fire patrol",
            "Set up monitoring perimeter and wind direction tracking"
        ],
        "Low": [
            "Dispatch local forestry officer for spot-check"
        ],
        "Very Low": [
            "Monitor satellite heat thermal anomaly data"
        ]
    },
    "Landslide": {
        "Critical": [
            "Deploy heavy excavation and earthmoving machinery immediately",
            "Dispatch specialized geological safety team to assess slope stability",
            "Send Search and Rescue teams with thermal imaging & K9 units",
            "Evacuate downhill and vulnerable structures immediately"
        ],
        "High": [
            "Deploy heavy excavation machinery and rescue team",
            "Assess secondary landslide risk on slope",
            "Set up emergency shelter for displaced residents"
        ],
        "Medium": [
            "Dispatch road clearance crew and rescue patrol",
            "Block unstable roadways to public traffic"
        ],
        "Low": [
            "Inspect slope stability and clear minor road debris"
        ],
        "Very Low": [
            "Log slope report and notify highway maintenance"
        ]
    },
    "Drought": {
        "Critical": [
            "Deploy emergency water tanker fleets to affected communities",
            "Distribute emergency food supplies and livestock feed",
            "Establish mobile health clinics for dehydration and malnutrition",
            "Activate emergency groundwater distribution networks"
        ],
        "High": [
            "Deploy water tankers and emergency relief supplies",
            "Monitor public drinking water reserves",
            "Set up relief distribution points"
        ],
        "Medium": [
            "Schedule regular water supply tankers",
            "Conduct community health and hydration checks"
        ],
        "Low": [
            "Monitor local reservoir levels and crop conditions"
        ],
        "Very Low": [
            "Track regional rainfall metrics and soil moisture"
        ]
    }
}


def get_response_recommendations(disaster_type: str, priority_level: str) -> List[str]:
    """
    Generate targeted rescue response action suggestions based on disaster type and priority level.
    Model-agnostic fallback included for any unexpected or unmapped disaster types.
    """
    disaster_dict = RECOMMENDATIONS_MAP.get(disaster_type)
    if disaster_dict:
        recs = disaster_dict.get(priority_level)
        if recs:
            return recs
        # Fall back to Medium if exact priority band is not mapped
        return disaster_dict.get("Medium", ["Deploy emergency assessment unit", "Contact local first responders"])
    
    # Generic fallback for unmapped disaster types
    return [
        "Deploy emergency first responder team to assess situation",
        "Establish field communication link with command center",
        "Coordinate with local emergency services for medical & rescue support"
    ]
