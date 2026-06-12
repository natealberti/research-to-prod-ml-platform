from fastapi import FastAPI

app = FastAPI(title="Research-to-Production ML Platform")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}