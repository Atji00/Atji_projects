from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class MaritalStatus(str, Enum):
    MARRIED = "married"
    DIVORCED = "divorced"
    SINGLE = "single"


class PreviousOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    OTHER = "other"
    UNKNOWN = "unknown"



class ClientInput(BaseModel):

    # Supprime les espaces en debut / fin de chaque champ texte (saisie formulaire)
    model_config = ConfigDict(str_strip_whitespace=True)

    # Client information not required for scoring  for another purpose
    name: str|None = Field(None, description="Nom du client", min_length=2, max_length=100)
    email: EmailStr|None = Field(None, description="Adresse email du client", max_length=50)

    # Required inputs for scoring model
    balance: float = Field(..., description="Solde du compte")
    campaign: int = Field(..., ge=1, description="Nombre de contacts pendant la campagne")
    previous: int = Field(..., ge=0, description="Nombre de contacts précédents")

    marital: MaritalStatus = Field(..., description="État matrimonial")
    housing: bool = Field(..., description="Prêt immobilier")
    loan: bool = Field(..., description="Prêt personnel")
    poutcome: PreviousOutcome = Field(..., description="Résultat de la campagne précédente")

    # Nom et email sont optionnels : un champ vide ("" ou espaces) devient None
    # au lieu d'echouer sur min_length / format email
    @field_validator("name", "email", mode="before")
    @classmethod
    def empty_to_none(cls, value):

        if isinstance(value, str) and not value.strip():
            return None

        return value


class PredictionResponse(BaseModel):

    prediction: Literal[0, 1] = Field(..., description="Classe prédite par le modèle")
    
    probability: float = Field(...,ge=0, le=1,description="Probabilité prédite pour la classe positive")