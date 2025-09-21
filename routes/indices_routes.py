from fastapi import APIRouter

def create_indices_routes(nse_client):
    router = APIRouter(prefix="/api/v1/indices", tags=["indices"])
    
    @router.get("/live-data")
    def get_all_indices():
        return nse_client.get_all_indices()
    
    return router