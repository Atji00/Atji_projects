from fastapi import Request


# Affichage de messages flash (erreur / succes / info) sur la page suivante
def flash(request: Request, message: str, category: str = "info") -> None:

    request.session.setdefault("_flashes", []).append(
        {"message": message, "category": category}
    )

# Recupere puis supprime les messages de la session
def pop_flashes(request: Request) -> list[dict]:

    return request.session.pop("_flashes", [])