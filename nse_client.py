import requests
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class NSEClient:
    def __init__(self):
        self.base_url = os.getenv("NSE_BASE_URL", "https://www.nseindia.com")
        self.timeout = int(os.getenv("NSE_TIMEOUT", "30"))
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
            'Referer': 'https://www.nseindia.com/market-data/52-week-high-equity-market',
            'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
    
    def _get_fresh_session(self):
        """Create fresh session with cookies for each API call"""
        session = requests.Session()
        session.headers.update(self.headers)
        try:
            session.get(f"{self.base_url}/market-data/52-week-high-equity-market", timeout=self.timeout)
        except:
            pass
        return session
    
    def get_quote_derivative(self, symbol: str) -> Dict[str, Any]:
        """Fetch derivative quote data including lot size"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/quote-derivative?symbol={symbol.upper()}",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()
    
    def get_52week_high_stocks(self) -> Dict[str, Any]:
        """Fetch 52-week high stock data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-52weekhighstock",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()
    
    def get_52week_high_stocks_data(self) -> Dict[str, Any]:
        """Fetch detailed 52-week high stock data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-data-52weekhighstock",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_52week_low_stocks_data(self) -> Dict[str, Any]:
        """Fetch detailed 52-week low stock data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-data-52weeklowstock",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_gainers_data(self) -> Dict[str, Any]:
        """Fetch gainers data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-variations?index=gainers",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_losers_data(self) -> Dict[str, Any]:
        """Fetch losers data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-variations?index=loosers",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_most_active_securities(self) -> Dict[str, Any]:
        """Fetch most active securities by value"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-most-active-securities?index=value",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_most_active_sme(self) -> Dict[str, Any]:
        """Fetch most active SME by volume"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-most-active-sme?index=volume",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_sec_gainers(self) -> Dict[str, Any]:
        """Fetch security gainers"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-variations?index=gainers&key=SecGtr20",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_volume_gainers(self) -> Dict[str, Any]:
        """Fetch volume gainers"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-volume-gainers",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_price_band_hitter(self) -> Dict[str, Any]:
        """Fetch price band hitter data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-price-band-hitter",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_advance_decline(self) -> Dict[str, Any]:
        """Fetch advance decline data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-advance",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_unchanged_data(self) -> Dict[str, Any]:
        """Fetch unchanged data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-unchanged",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_decline_data(self) -> Dict[str, Any]:
        """Fetch decline data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-decline",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_stocks_traded(self) -> Dict[str, Any]:
        """Fetch stocks traded data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-stocksTraded",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_large_deals(self) -> Dict[str, Any]:
        """Fetch large deals data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-large-deals",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_all_indices(self) -> Dict[str, Any]:
        """Fetch all indices data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/allIndices",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_derivatives_snapshot(self) -> Dict[str, Any]:
        """Fetch derivatives market snapshot"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-derivatives",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_most_active_underlying(self) -> Dict[str, Any]:
        """Fetch most active underlying"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-most-active-underlying",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_oi_spurts_underlyings(self) -> Dict[str, Any]:
        """Fetch OI spurts underlyings"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-oi-spurts-underlyings",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_oi_spurts_contracts(self) -> Dict[str, Any]:
        """Fetch OI spurts contracts"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-oi-spurts-contracts",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_option_chain_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch option chain info for symbol"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/option-chain-indices?symbol={symbol.upper()}",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_option_chain(self, symbol: str, expiry: str = None) -> Dict[str, Any]:
        """Fetch option chain data"""
        session = self._get_fresh_session()
        try:
            url = f"{self.base_url}/api/option-chain-indices?symbol={symbol.upper()}"
            if expiry:
                url += f"&expiry={expiry}"
            response = session.get(url, timeout=10)
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()
    
    def get_index_constituents(self, index_name: str) -> Dict[str, Any]:
        """Fetch index constituents data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/equity-stockIndices?index={index_name.upper()}",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()

    def get_derivatives_equity_snapshot(self, limit: int = 20) -> Dict[str, Any]:
        """Fetch derivatives equity snapshot data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/snapshot-derivatives-equity?index=contracts&limit={limit}",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()
    
    def get_top20_derivatives_contracts(self) -> Dict[str, Any]:
        """Fetch top 20 derivatives contracts data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/liveEquity-derivatives?index=top20_contracts",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()
    
    def get_most_active_underlying(self) -> Dict[str, Any]:
        """Fetch most active underlying derivatives data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/live-analysis-most-active-underlying",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()
    
    def get_futures_master_quote(self) -> Dict[str, Any]:
        """Fetch futures master quote data - all future stock symbols"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/master-quote",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()
    
    def get_historical_data(self, symbol: str, from_date: str, to_date: str) -> Dict[str, Any]:
        """Fetch historical data for symbol"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/historicalOR/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={from_date}&to={to_date}",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()
    
    def get_options_historical_data(self, symbol: str, option_type: str, strike_price: float, expiry_date: str, from_date: str, to_date: str) -> Dict[str, Any]:
        """Fetch options historical data"""
        session = self._get_fresh_session()
        try:
            response = session.get(
                f"{self.base_url}/api/historicalOR/fo/derivatives?from={from_date}&to={to_date}&optionType={option_type}&strikePrice={strike_price}&expiryDate={expiry_date}&instrumentType=OPTIDX&symbol={symbol}",
                timeout=10
            )
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            else:
                return {"error": "Empty response from NSE", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e), "data": None}
        finally:
            session.close()