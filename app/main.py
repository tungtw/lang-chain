from fastapi import FastAPI
from app.config import settings

# Initialize FastAPI app with settings
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.app_name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)