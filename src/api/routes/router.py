from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from api.modules.flash_messages import flash, pop_flashes
from api.modules.prediction_artefacts import Artefacts
from api.modules.pydantic_scheme import (
    ClientInput,
    MaritalStatus,
    PreviousOutcome,
)
from scoring_model.runtime.paths import ProjectPaths

TEMPLATES_DIR = ProjectPaths().templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(tags=["predict"])


@router.get("/", response_class=HTMLResponse, name="form_page")
def form_page(request: Request):

    # Affiche le formulaire ; en cas d'erreur precedente, les valeurs saisies
    # sont re-injectees pour eviter a l'utilisateur de tout ressaisir
    return templates.TemplateResponse(
        request,
        "clientInformations.html",
        {
            "flashes": pop_flashes(request),
            "values": request.session.pop("form_values", {}),
            "marital_options": [m.value for m in MaritalStatus],
            "poutcome_options": [p.value for p in PreviousOutcome],
        },
    )


@router.post("/predict", response_class=RedirectResponse, name="submit_form")
def submit_form(request: Request, client: Annotated[ClientInput, Form()]):

    values = client.model_dump(mode="json")

    try:

        result = Artefacts(client).run_prediction()

    except HTTPException as error:

        flash(request, str(error.detail), "error")
        request.session["form_values"] = values

        return RedirectResponse(str(request.url_for("form_page")), status_code=303)


    request.session["result"] = {
                                "prediction": result.prediction,
                                "probability": result.probability,
                                "client": values,
                                }

    flash(request, "Prédiction calculée avec succès.", "success")

    return RedirectResponse(str(request.url_for("result_page")), status_code=303)


@router.get("/result", response_class=HTMLResponse, name="result_page")
def result_page(request: Request):

    # Le resultat est consomme a la lecture (affiche une seule fois)
    result = request.session.pop("result", None)

    client = ClientInput(**result["client"]) if result else None
    # Acces direct a /result sans avoir soumis le formulaire
    if result is None:

        flash(request, "Aucune prédiction à afficher : remplissez le formulaire.", "info")

        return RedirectResponse(str(request.url_for("form_page")), status_code=303)

    return templates.TemplateResponse(
        request,
        "clientPredictionResults.html",
        {
            "flashes": pop_flashes(request),
            "result": result,
            "waterfall": Artefacts(client).plot_waterfall(),
            "probability_pct": round(result["probability"] * 100, 2),
        }
    )