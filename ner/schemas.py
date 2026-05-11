"""
Pydantic schema for structured patient profile.
Output format shared by both NER approaches (LLM and BioBERT).
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class PatientProfile(BaseModel):
    age: Optional[int] = Field(None, description="Patient age in years")
    gender: Optional[str] = Field(None, description="male, female, or other")

    conditions: List[str] = Field(default_factory=list, description="Medical conditions and diagnoses")
    medications: List[str] = Field(default_factory=list, description="Current medications with dosage if mentioned")
    lab_values: Dict[str, str] = Field(default_factory=dict, description="Lab results e.g. {'HbA1c': '8.2%'}")
    prior_treatments: List[str] = Field(default_factory=list, description="Past treatments, surgeries, procedures")
    devices: List[str] = Field(default_factory=list, description="Medical devices e.g. pacemaker")

    location_city: Optional[str] = Field(None, description="Patient city")
    location_state: Optional[str] = Field(None, description="2-letter state abbreviation e.g. IL")
    travel_distance_miles: Optional[int] = Field(None, description="Max miles willing to travel")

    healthy_volunteer: bool = Field(False, description="True if no conditions and wants to participate as healthy volunteer")
    disease_status: Optional[str] = Field(None, description="remission, active, stable, newly diagnosed")

    # Populated from lab report upload + diagnostic reasoning
    biomarkers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Biomarker values extracted from lab report"
    )
    diagnostic_suggestions: List[str] = Field(
        default_factory=list,
        description="Conditions suggested by diagnostic reasoning agent"
    )
    trial_relevant_values: List[str] = Field(
        default_factory=list,
        description="Biomarker values specifically relevant for trial eligibility"
    )

    # Populated after extraction, not from patient text
    missing_fields: List[str] = Field(default_factory=list, description="Critical fields still needed")

    def to_query_string(self) -> str:
        """Convert profile to natural language query for retrieval."""
        parts = []
        if self.conditions:
            parts.append(f"Conditions: {', '.join(self.conditions)}")
        if self.diagnostic_suggestions:
            parts.append(f"Suggested diagnoses: {', '.join(self.diagnostic_suggestions)}")
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.gender:
            parts.append(f"Gender: {self.gender}")
        if self.medications:
            parts.append(f"Medications: {', '.join(self.medications)}")
        if self.lab_values:
            parts.append(f"Lab values: {', '.join(f'{k}: {v}' for k, v in self.lab_values.items())}")
        if self.biomarkers:
            bm_parts = []
            for name, data in self.biomarkers.items():
                if isinstance(data, dict):
                    val = data.get("value", "")
                    unit = data.get("unit", "")
                    status = data.get("status", "")
                    bm_parts.append(f"{name}: {val}{unit} ({status})")
            if bm_parts:
                parts.append(f"Biomarkers: {', '.join(bm_parts)}")
        if self.trial_relevant_values:
            parts.append(f"Trial-relevant findings: {'; '.join(self.trial_relevant_values)}")
        if self.location_city:
            parts.append(f"Location: {self.location_city}, {self.location_state or ''}")
        if self.disease_status:
            parts.append(f"Disease status: {self.disease_status}")
        return ". ".join(parts)

    def get_missing_critical_fields(self) -> List[str]:
        """Return fields we critically need but don't have yet."""
        missing = []
        if not self.conditions and not self.healthy_volunteer:
            missing.append("medical conditions or diagnoses")
        if not self.age:
            missing.append("age")
        if not self.gender:
            missing.append("gender")
        if not self.location_city:
            missing.append("location (city or zip code)")
        return missing
