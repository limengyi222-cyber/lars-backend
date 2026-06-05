from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import json
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
    try:
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        url = f"https://opendata.adsb.fi/api/v2/lat/{center_lat}/lon/{center_lon}/dist/250"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=20)

        if resp.status_code != 200:
            return {"flights": [], "error": f"adsb.fi returned {resp.status_code}", "total": 0}

        aircraft = resp.json().get("aircraft") or []
        flights = []
        for ac in aircraft:
            lat = ac.get("lat")
            lon = ac.get("lon")
            if lat is None or lon is None:
                continue
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
            alt = ac.get("alt_baro") or ac.get("alt_geom")
            on_ground = isinstance(alt, str) and alt == "ground"
            flights.append({
                "icao24":        ac.get("hex", ""),
                "callsign":      (ac.get("flight") or "").strip(),
                "longitude":     lon,
                "latitude":      lat,
                "altitude":      None if on_ground else alt,
                "on_ground":     on_ground,
                "velocity":      ac.get("gs"),
                "heading_deg":   ac.get("track"),
                "vertical_rate": ac.get("baro_rate"),
            })

        return {"flights": flights, "total": len(flights)}

    except Exception as e:
        return {"flights": [], "error": f"{type(e).__name__}: {e}", "total": 0}


_GRIDS_CACHE = None

@app.get("/api/v1/airspace/grids")
async def airspace_grids():
    global _GRIDS_CACHE
    if _GRIDS_CACHE is None:
        grid_path = os.path.join(os.path.dirname(__file__), "gba_grids.json")
        with open(grid_path, "r") as f:
            _GRIDS_CACHE = f.read()
    return JSONResponse(content=json.loads(_GRIDS_CACHE))


@app.get("/health")
async def health():
    return {"status": "ok"}
