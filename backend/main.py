"""
VideoRAG Backend - FastAPI Application
Handles authentication and video processing pipeline
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.routes import router as auth_router
from video.routes import router as video_router

app = FastAPI(
    title="VideoRAG API",
    description="Backend for VideoRAG - Video Retrieval Augmented Generation",
    version="1.0.0"
)

# CORS - allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(video_router, prefix="/api/video", tags=["Video Processing"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "VideoRAG Backend"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VideoRAG API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
