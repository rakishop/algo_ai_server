from fastapi import APIRouter

def create_derivatives_routes(nse_client):
    router = APIRouter(prefix="/api/v1/derivatives", tags=["derivatives"])
    
    @router.get("/market-snapshot")
    def get_derivatives_snapshot():
        return nse_client.get_derivatives_snapshot()

    @router.get("/active-underlyings")
    def get_most_active_underlying():
        return nse_client.get_most_active_underlying()

    @router.get("/open-interest-spurts")
    def get_oi_spurts():
        underlyings = nse_client.get_oi_spurts_underlyings()
        contracts = nse_client.get_oi_spurts_contracts()
        return {
            "underlyings": underlyings,
            "contracts": contracts
        }
    
    return router