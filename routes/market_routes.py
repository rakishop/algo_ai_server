from fastapi import APIRouter

def create_market_routes(nse_client):
    router = APIRouter(prefix="/api/v1/market", tags=["market"])
    
    @router.get("/52-week-extremes")
    def get_52week_high_stocks_data():
        high_data = nse_client.get_52week_high_stocks_data()
        low_data = nse_client.get_52week_low_stocks_data() 
        analysis_response = nse_client.get_52week_high_stocks()
        return {
            "52_week_high": high_data,
            "52_week_low": low_data,
            "analysis_api": analysis_response
        }

    @router.get("/daily-movers")
    def get_gainers_losers():
        gainers = nse_client.get_gainers_data()
        losers = nse_client.get_losers_data()
        return {
            "gainers": gainers,
            "losers": losers
        }

    @router.get("/activity-summary")
    def get_market_activity():
        most_active = nse_client.get_most_active_securities()
        active_sme = nse_client.get_most_active_sme()
        sec_gainers = nse_client.get_sec_gainers()
        volume_gainers = nse_client.get_volume_gainers()
        return {
            "most_active_securities": most_active,
            "most_active_sme": active_sme,
            "security_gainers": sec_gainers,
            "volume_gainers": volume_gainers
        }

    @router.get("/price-band-hits")
    def get_price_band_hitter():
        return nse_client.get_price_band_hitter()

    @router.get("/volume-leaders")
    def get_volume_gainers():
        return nse_client.get_volume_gainers()

    @router.get("/breadth-indicators")
    def get_advance_decline():
        advance = nse_client.get_advance_decline()
        decline = nse_client.get_decline_data()
        unchanged = nse_client.get_unchanged_data()
        return {
            "advance": advance,
            "decline": decline,
            "unchanged": unchanged
        }

    @router.get("/trading-statistics")
    def get_stocks_traded():
        return nse_client.get_stocks_traded()

    @router.get("/block-deals")
    def get_large_deals():
        return nse_client.get_large_deals()
    
    return router