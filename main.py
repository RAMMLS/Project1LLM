from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router

app = FastAPI(
    title="Unified AI Lab & Analysis Platform",
    description="Professional web application backend for standardizing different neural network tasks.",
    version="1.0.0"
)

# Configure CORS for Frontend integration (React/Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main router
app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Unified AI Lab & Analysis Platform API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
