"""
Tactical Scenario Fixtures for E2E Testing.

Provides real-world tactical and cybersecurity scenarios
for comprehensive end-to-end testing of the multi-agent system.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TacticalScenario:
    """Represents a tactical decision scenario."""

    id: str
    name: str
    domain: str
    query: str
    initial_state: Dict[str, Any]
    possible_actions: List[str]
    expected_outcome: Dict[str, Any]
    difficulty: str
    constraints: List[str]


# Military Tactical Scenarios
MILITARY_DEFENSIVE_POSITION = TacticalScenario(
    id="mil_defense_001",
    name="Northern Perimeter Defense",
    domain="military",
    query=(
        "Enemy forces spotted approaching from north. Night conditions with limited visibility. "
        "Available assets: Infantry platoon (25 personnel), UAV support, limited ammunition (100 rounds per soldier). "
        "Recommend optimal defensive position and engagement strategy."
    ),
    initial_state={
        "position": "neutral",
        "resources": {"ammo": 2500, "fuel": 80, "personnel": 25},
        "enemy_position": "north",
        "enemy_strength": "unknown",
        "visibility": "low",
        "time_of_day": "night",
        "terrain": "mixed_urban_rural",
        "communication": "operational",
        "reinforcement_eta_hours": 6,
    },
    possible_actions=[
        "advance_to_alpha",
        "hold_current_position",
        "retreat_to_beta",
        "flanking_maneuver",
        "request_reinforcement",
        "establish_observation_posts",
    ],
    expected_outcome={
        "recommended_action": "establish_observation_posts",
        "confidence_range": (0.70, 0.85),
        "key_risks": ["ammo_consumption", "uav_vulnerability", "enemy_reinforcement"],
        "expected_casualties": (0, 5),
    },
    difficulty="hard",
    constraints=["limited_ammo", "night_conditions", "unknown_enemy_strength"],
)

MILITARY_RECONNAISSANCE = TacticalScenario(
    id="mil_recon_001",
    name="Urban Reconnaissance Mission",
    domain="military",
    query=(
        "Tasked with reconnaissance of urban area. Intelligence suggests possible enemy presence. "
        "Team of 6 operators with stealth equipment. Mission: Identify enemy positions without detection. "
        "Time constraint: 4 hours before extraction window."
    ),
    initial_state={
        "team_size": 6,
        "equipment": ["night_vision", "thermal_imaging", "drones", "suppressors"],
        "target_area": "urban_center",
        "enemy_presence": "suspected",
        "extraction_window_hours": 4,
        "communication": "encrypted",
    },
    possible_actions=[
        "split_team_coverage",
        "maintain_single_unit",
        "drone_first_sweep",
        "ground_infiltration",
        "establish_overwatch",
        "abort_mission",
    ],
    expected_outcome={
        "recommended_action": "drone_first_sweep",
        "confidence_range": (0.75, 0.90),
        "key_risks": ["detection", "communication_intercept", "time_constraint"],
    },
    difficulty="medium",
    constraints=["stealth_required", "time_limited", "no_engagement"],
)

MILITARY_RESOURCE_ALLOCATION = TacticalScenario(
    id="mil_resource_001",
    name="Multi-Front Resource Allocation",
    domain="military",
    query=(
        "Command center overseeing 3 active fronts. Limited reserves available for reinforcement. "
        "Front A: Heavy engagement, casualty rate high. Front B: Moderate engagement, strategic position. "
        "Front C: Light engagement, supply line protection. "
        "Allocate 2 reserve units optimally."
    ),
    initial_state={
        "reserve_units": 2,
        "fronts": {
            "A": {"engagement": "heavy", "casualties": "high", "strategic_value": 0.6},
            "B": {"engagement": "moderate", "casualties": "medium", "strategic_value": 0.9},
            "C": {"engagement": "light", "casualties": "low", "strategic_value": 0.7},
        },
        "overall_objective": "maintain_strategic_positions",
    },
    possible_actions=[
        "all_to_front_a",
        "all_to_front_b",
        "all_to_front_c",
        "split_a_and_b",
        "split_b_and_c",
        "split_a_and_c",
    ],
    expected_outcome={
        "recommended_action": "split_b_and_c",
        "rationale": "Secure strategic position and protect supply lines",
        "confidence_range": (0.65, 0.80),
    },
    difficulty="hard",
    constraints=["limited_reserves", "multiple_objectives", "uncertain_enemy_intent"],
)

# Cybersecurity Threat Scenarios
CYBER_APT_INTRUSION = TacticalScenario(
    id="cyber_apt_001",
    name="APT28 Infrastructure Intrusion",
    domain="cybersecurity",
    query=(
        "APT28 indicators detected on critical infrastructure network. "
        "Evidence of credential harvesting (T1003) and lateral movement (T1021). "
        "3 systems compromised, C2 communication established. "
        "Recommend immediate containment and response actions while preserving forensic evidence."
    ),
    initial_state={
        "threat_actor": "APT28",
        "threat_level": "critical",
        "systems_compromised": 3,
        "mitre_techniques": ["T1003", "T1021", "T1071"],
        "c2_active": True,
        "evidence_available": ["network_logs", "endpoint_data", "memory_dumps"],
        "affected_assets": ["domain_controller", "file_server", "workstation"],
    },
    possible_actions=[
        "isolate_all_systems",
        "selective_isolation",
        "collect_forensics_first",
        "block_c2_communication",
        "credential_reset",
        "notify_authorities",
        "activate_incident_response_team",
    ],
    expected_outcome={
        "primary_action": "block_c2_communication",
        "secondary_actions": ["selective_isolation", "collect_forensics_first"],
        "confidence_range": (0.80, 0.92),
        "key_risks": ["evidence_destruction", "threat_persistence", "data_exfiltration"],
    },
    difficulty="critical",
    constraints=["preserve_evidence", "stop_active_threat", "minimize_downtime"],
)

CYBER_RANSOMWARE_INCIDENT = TacticalScenario(
    id="cyber_ransom_001",
    name="Ransomware Attack Response",
    domain="cybersecurity",
    query=(
        "Ransomware detected spreading across network. Initial infection via phishing email. "
        "15 endpoints encrypted, file servers at risk. Backup status: Last verified 48 hours ago. "
        "Encryption actively spreading. Recommend containment and recovery strategy."
    ),
    initial_state={
        "incident_type": "ransomware",
        "infection_vector": "phishing",
        "endpoints_encrypted": 15,
        "total_endpoints": 500,
        "file_servers_at_risk": 3,
        "backup_age_hours": 48,
        "backup_verified": True,
        "encryption_speed": "fast_spreading",
    },
    possible_actions=[
        "network_isolation",
        "shutdown_file_servers",
        "disconnect_affected_endpoints",
        "engage_backup_restoration",
        "contact_ransomware_negotiators",
        "deploy_decryption_tools",
        "rebuild_from_backup",
    ],
    expected_outcome={
        "immediate_action": "network_isolation",
        "recovery_strategy": "rebuild_from_backup",
        "confidence_range": (0.85, 0.95),
        "expected_recovery_hours": 24,
    },
    difficulty="high",
    constraints=["stop_spread", "data_recovery", "business_continuity"],
)

CYBER_INSIDER_THREAT = TacticalScenario(
    id="cyber_insider_001",
    name="Insider Threat Detection",
    domain="cybersecurity",
    query=(
        "Anomalous data access patterns detected for privileged user. "
        "Large file downloads outside business hours, access to sensitive repositories. "
        "User scheduled to leave company in 2 weeks. "
        "Investigate and contain potential data exfiltration."
    ),
    initial_state={
        "suspect_type": "insider",
        "user_status": "employed_leaving",
        "anomalous_behavior": ["large_downloads", "off_hours_access", "sensitive_repos"],
        "time_until_departure_days": 14,
        "evidence_strength": "moderate",
        "legal_constraints": True,
    },
    possible_actions=[
        "revoke_access_immediately",
        "increase_monitoring",
        "conduct_interview",
        "legal_consultation",
        "forensic_investigation",
        "dlp_policy_enforcement",
        "terminate_employment",
    ],
    expected_outcome={
        "recommended_approach": "increase_monitoring",
        "parallel_action": "legal_consultation",
        "confidence_range": (0.70, 0.85),
        "legal_risk": "medium",
    },
    difficulty="high",
    constraints=["legal_requirements", "evidence_preservation", "employee_rights"],
)

# Data Analysis Scenarios (for DABStep-style multi-step reasoning)
DATA_ANALYSIS_CORRELATION = TacticalScenario(
    id="data_analysis_001",
    name="Multi-Variable Correlation Analysis",
    domain="data_analysis",
    query=(
        "Dataset contains 10,000 records with 15 variables. "
        "Objective: Identify key factors influencing customer churn. "
        "Variables include: usage_frequency, support_tickets, subscription_length, "
        "payment_delays, feature_adoption, competitor_comparison. "
        "Provide step-by-step analysis approach."
    ),
    initial_state={
        "dataset_size": 10000,
        "variables": 15,
        "target_variable": "churn",
        "missing_data_percentage": 5,
        "outliers_detected": True,
    },
    possible_actions=[
        "exploratory_data_analysis",
        "correlation_matrix",
        "feature_importance_ranking",
        "regression_analysis",
        "clustering_analysis",
        "time_series_decomposition",
    ],
    expected_outcome={
        "analysis_steps": [
            "exploratory_data_analysis",
            "correlation_matrix",
            "feature_importance_ranking",
        ],
        "key_factors": ["support_tickets", "payment_delays", "feature_adoption"],
        "confidence_range": (0.75, 0.90),
    },
    difficulty="medium",
    constraints=["missing_data", "multicollinearity", "interpretability"],
)

# Collection of all scenarios
ALL_TACTICAL_SCENARIOS = [
    MILITARY_DEFENSIVE_POSITION,
    MILITARY_RECONNAISSANCE,
    MILITARY_RESOURCE_ALLOCATION,
    CYBER_APT_INTRUSION,
    CYBER_RANSOMWARE_INCIDENT,
    CYBER_INSIDER_THREAT,
    DATA_ANALYSIS_CORRELATION,
]

MILITARY_SCENARIOS = [
    MILITARY_DEFENSIVE_POSITION,
    MILITARY_RECONNAISSANCE,
    MILITARY_RESOURCE_ALLOCATION,
]

CYBERSECURITY_SCENARIOS = [
    CYBER_APT_INTRUSION,
    CYBER_RANSOMWARE_INCIDENT,
    CYBER_INSIDER_THREAT,
]

DATA_ANALYSIS_SCENARIOS = [
    DATA_ANALYSIS_CORRELATION,
]


def get_scenario_by_id(scenario_id: str) -> Optional[TacticalScenario]:
    """Get scenario by ID."""
    for scenario in ALL_TACTICAL_SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    return None


def get_scenarios_by_domain(domain: str) -> List[TacticalScenario]:
    """Get all scenarios for a specific domain."""
    return [s for s in ALL_TACTICAL_SCENARIOS if s.domain == domain]


def get_scenarios_by_difficulty(difficulty: str) -> List[TacticalScenario]:
    """Get all scenarios of specific difficulty."""
    return [s for s in ALL_TACTICAL_SCENARIOS if s.difficulty == difficulty]
