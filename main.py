from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

OPENSKY_USER = os.getenv("OPENSKY_USER", "lncyi")
OPENSKY_PASS = os.getenv("OPENSKY_PASS", "")


@app.get("/api/v1/flights/live")
async def live_flights(
    min_lon: float = 112.5,
    min_lat: float = 22.0,
    max_lon: float = 115.0,
    max_lat: float = 23.5,
):
    url = "https://opensky-network.org/api/states/all"
    params = {"lamin": min_lat, "lomin": min_lon, "lamax": max_lat, "lomax": max_lon}

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url,
            params=params,
            auth=(OPENSKY_USER, OPENSKY_PASS),
            timeout=15,
        )

    if resp.status_code != 200:
        return {"flights": [], "error": f"OpenSky returned {resp.status_code}"}

    states = resp.json().get("states") or []
    flights = []
    for s in states:
        if s[5] is None or s[6] is None:
            continue
        flights.append({
            "icao24":        s[0],
            "callsign":      (s[1] or "").strip(),
            "longitude":     s[5],
            "latitude":      s[6],
            "altitude":      s[7],
            "on_ground":     s[8],
            "velocity":      s[9],
            "heading_deg":   s[10],
            "vertical_rate": s[11],
        })

    return {"flights": flights, "total": len(flights)}


@app.get("/health")
async def health():
    return {"status": "ok"}
