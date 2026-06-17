"""Confederation priors and feature helpers."""
from typing import Dict

CONFEDERATION_ATTACK_PRIOR: Dict[str, float] = {
    "UEFA": 0.10,
    "CONMEBOL": 0.15,
    "CONCACAF": -0.10,
    "CAF": -0.05,
    "AFC": -0.10,
    "OFC": -0.30,
}

TEAM_CONFEDERATION: Dict[str, str] = {
    # CONMEBOL
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Peru": "CONMEBOL", "Venezuela": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    # UEFA
    "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA",
    "Italy": "UEFA", "Croatia": "UEFA", "Denmark": "UEFA",
    "Switzerland": "UEFA", "Austria": "UEFA", "Poland": "UEFA",
    "Serbia": "UEFA", "Czech Republic": "UEFA", "Hungary": "UEFA",
    "Romania": "UEFA", "Scotland": "UEFA", "Ukraine": "UEFA",
    "Turkey": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Albania": "UEFA", "Georgia": "UEFA", "Greece": "UEFA",
    "Norway": "UEFA", "Sweden": "UEFA", "Finland": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "North Macedonia": "UEFA",
    "Montenegro": "UEFA", "Iceland": "UEFA", "Wales": "UEFA",
    "Kosovo": "UEFA", "Luxembourg": "UEFA",
    # CONCACAF
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Panama": "CONCACAF", "Jamaica": "CONCACAF",
    "Honduras": "CONCACAF", "El Salvador": "CONCACAF", "Haiti": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF", "Cuba": "CONCACAF",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF",
    "Ivory Coast": "CAF", "Ghana": "CAF", "Cameroon": "CAF",
    "Egypt": "CAF", "Tunisia": "CAF", "South Africa": "CAF",
    "Mali": "CAF", "Burkina Faso": "CAF", "Algeria": "CAF",
    "DR Congo": "CAF", "Guinea": "CAF", "Zambia": "CAF",
    "Tanzania": "CAF", "Cape Verde": "CAF", "Gabon": "CAF",
    # AFC
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC",
    "Saudi Arabia": "AFC", "Australia": "AFC", "Qatar": "AFC",
    "UAE": "AFC", "Iraq": "AFC", "China PR": "AFC",
    "Uzbekistan": "AFC", "Indonesia": "AFC", "Jordan": "AFC",
    "Oman": "AFC", "Bahrain": "AFC", "Kuwait": "AFC",
    # OFC
    "New Zealand": "OFC",
}


def confederation(team: str) -> str:
    return TEAM_CONFEDERATION.get(team, "UEFA")


def attack_prior(team: str) -> float:
    return CONFEDERATION_ATTACK_PRIOR.get(confederation(team), 0.0)
