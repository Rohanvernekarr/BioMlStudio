"""
API Routes for BioMLStudio

This package contains all the API route handlers organized by functionality.
"""

from fastapi import APIRouter

from . import auth, datasets, health, jobs, models

# Create main router that includes all sub-routes
router = APIRouter()

# Include all route modules
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
router.include_router(models.router, prefix="/models", tags=["models"])

__all__ = ["router", "auth", "datasets", "health", "jobs", "models"]
