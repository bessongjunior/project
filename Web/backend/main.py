from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="SmartPick Logistics Simulation", version="0.1.0")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "Operational", "active_cities": ["Chongqing", "Shanghai"]}

@router.post("/optimize")
async def optimize_route(data: dict):
    # This will call our GCN Inference Service
    return {"optimized_sequence": [1, 5, 2], "estimated_eta": 45.5}

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
