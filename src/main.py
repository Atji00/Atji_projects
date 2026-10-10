from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.middleware.errors_handling import setup_middleware
from api.routes.router import router
from scoring_model.runtime.paths import ProjectPaths

load_dotenv()

STATIC_DIR = ProjectPaths().static

app = FastAPI(title="Credit Scoring Modelling API", version="1.0.0")

# Session, gestion des erreurs de validation du formulaire
setup_middleware(app)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.include_router(router)

if __name__ == "__main__":

    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
