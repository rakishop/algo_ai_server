import requests
import re
from config import settings
from ai_technical_analyzer import AITechnicalAnalyzer
from nse_client import NSEClient
from market_scanner import MarketScanner

class TelegramHandler:
    def __init__(self):
        self.nse_client = NSEClient()
        self.ai_analyzer = AITechnicalAnalyzer()
        self.scanner = MarketScanner()
        
    def extract_symbol(self, message):
        """Extract stock symbol from user message"""
        message = message.upper()
        
        # Common stock symbols with variations
        symbol_map = {
            'ITC': 'ITC', 'RELIANCE': 'RELIANCE', 'TCS': 'TCS', 'INFY': 'INFY', 'INFOSYS': 'INFY',
            'HDFC': 'HDFCBANK', 'HDFCBANK': 'HDFCBANK', 'ICICI': 'ICICIBANK', 'ICICIBANK': 'ICICIBANK',
            'SBI': 'SBIN', 'SBIN': 'SBIN', 'BHARTI': 'BHARTIARTL', 'AIRTEL': 'BHARTIARTL',
            'ASIAN': 'ASIANPAINT', 'ASIANPAINT': 'ASIANPAINT', 'MARUTI': 'MARUTI', 'SUZUKI': 'MARUTI',
            'KOTAK': 'KOTAKBANK', 'LT': 'LT', 'LARSEN': 'LT', 'AXIS': 'AXISBANK', 'AXISBANK': 'AXISBANK',
            'WIPRO': 'WIPRO', 'ULTRATECH': 'ULTRACEMCO', 'NESTLE': 'NESTLEIND', 'HCL': 'HCLTECH',
            'BAJAJ': 'BAJFINANCE', 'BAJFINANCE': 'BAJFINANCE', 'TITAN': 'TITAN'
        }
        
        # Check for stock-related keywords and extract symbol
        words = message.split()
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^A-Z]', '', word)
            if word_clean in symbol_map:
                return symbol_map[word_clean]
            
            # Check if word contains stock symbol
            for symbol, mapped in symbol_map.items():
                if symbol in word_clean:
                    return mapped
        
        return None
    
    def analyze_symbol(self, symbol):
        """Analyze symbol using AI models"""
        try:
            # Get market data
            active_data = self.nse_client.get_most_active_securities()
            volume_data = self.nse_client.get_volume_gainers()
            
            # Find symbol in data
            stock_data = None
            for data_source in [active_data, volume_data]:
                if data_source.get("data"):
                    for stock in data_source["data"]:
                        if stock.get("symbol") == symbol:
                            stock_data = stock
                            break
                if stock_data:
                    break
            
            if not stock_data:
                return f"❌ {symbol} not found in current market data"
            
            # AI Analysis
            analysis = self.ai_analyzer.analyze_stock(stock_data, [])
            
            # Format response
            price = stock_data.get('lastPrice', 0)
            change = stock_data.get('pChange', 0)
            volume = stock_data.get('quantityTraded', 0)
            
            ai_signal = analysis.get('ai_analysis', {}).get('predicted_signal', 'HOLD')
            confidence = analysis.get('ai_analysis', {}).get('confidence', 50)
            
            message = f"🤖 AI ANALYSIS: {symbol}\n\n"
            message += f"💰 Price: ₹{price:.1f}\n"
            message += f"📊 Change: {change:+.1f}%\n"
            message += f"📈 Volume: {volume:,}\n\n"
            message += f"🎯 AI Signal: {ai_signal}\n"
            message += f"🔥 Confidence: {confidence:.0f}%\n\n"
            
            if ai_signal == "BUY":
                message += "✅ AI recommends BUY"
            elif ai_signal == "SELL":
                message += "🔴 AI recommends SELL"
            else:
                message += "⚪ AI recommends HOLD"
                
            return message
            
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"
    
    def send_message(self, chat_id, message):
        """Send message to user"""
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        return requests.post(url, data=data)
    
    def handle_message(self, update):
        """Handle incoming message"""
        if 'message' not in update:
            return
            
        message = update['message']
        chat_id = message['chat']['id']
        text = message.get('text', '')
        
        # Only respond to messages that seem to be asking about stocks
        stock_keywords = ['TELL', 'ABOUT', 'STOCK', 'PRICE', 'ANALYSIS', 'BUY', 'SELL', 'WHAT', 'HOW']
        if not any(keyword in text.upper() for keyword in stock_keywords):
            return  # Ignore non-stock related messages
        
        # Extract symbol
        symbol = self.extract_symbol(text)
        if not symbol:
            response = "🤖 I can analyze stocks like ITC, RELIANCE, TCS, INFY, etc. Try: 'Tell me about ITC stock'"
            self.send_message(chat_id, response)
            return
        
        # Analyze and respond
        analysis = self.analyze_symbol(symbol)
        self.send_message(chat_id, analysis)