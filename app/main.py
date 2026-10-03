from fastapi import FastAPI


app = FastAPI(title="Kartly Hybrid Order Support Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
