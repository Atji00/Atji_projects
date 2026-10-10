import os

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse
from starlette.datastructures import FormData
from starlette.middleware.sessions import SessionMiddleware

from api.modules.flash_messages import flash

# Libelles des messages d'erreur affiches dans le formulaire HTML
FIELD_LABELS = {
                    "name": "Nom",
                    "email": "Email",
                    "balance": "Solde bancaire",
                    "campaign": "Nombre de contacts",
                    "previous": "Contacts précédents",
                    "marital": "Situation matrimoniale",
                    "housing": "Crédit immobilier",
                    "loan": "Prêt personnel",
                    "poutcome": "Résultat de la campagne précédente",
                }

# Route du formulaire dont les erreurs de validation sont renvoyees en flash
FORM_ROUTE = "submit_form"


async def validation_exception_handler(request: Request, error: RequestValidationError):

    # Hors formulaire : reponse 422 JSON standard de FastAPI
    route = request.scope.get("route")

    if getattr(route, "name", None) != FORM_ROUTE:

        return await request_validation_exception_handler(request, error)

    # Un message lisible par champ invalide (loc = ("body", "<champ>"))
    for err in error.errors():

        field = str(err["loc"][-1])
        label = FIELD_LABELS.get(field, field)
        flash(request, f"{label} : {err['msg']}", "error")

    # On conserve la saisie pour la re-afficher dans le formulaire
    if isinstance(error.body, FormData):

        request.session["form_values"] = {
            key: value for key, value in error.body.items() if isinstance(value, str)
        }

    return RedirectResponse(str(request.url_for("form_page")), status_code=303)


def setup_middleware(app: FastAPI) -> None:

    # Session signee par cookie : messages flash, valeurs du formulaire, resultat
    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv("SESSION_SECRET_KEY"),
    )

    # Erreurs de validation Pydantic du formulaire -> flash + redirection
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
