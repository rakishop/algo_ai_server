class AIStockSelector:
    def __init__(self):
        pass
    
    def calculate_ai_score(self, stock):
        """Calculate AI trading score with breakout detection"""
        try:
            price = float(stock.get('ltp', 0))
            change = float(stock.get('pChange', 0))
            volume = float(stock.get('totalTradedVolume', 0))
            value = float(stock.get('totalTradedValue', 0))
            
            # 15-min breakout detection
            is_15min_breakout = abs(change) > 2.5 and volume > 3000000
            
            # Daily breakout detection  
            is_daily_breakout = abs(change) > 4 and volume > 5000000
            
            # Only select breakout stocks
            if not (is_15min_breakout or is_daily_breakout):
                return 0
            
            # AI scoring for breakout stocks
            momentum_score = min(abs(change) * 3, 30)
            volume_score = min(volume / 1000000, 20)
            value_score = min(value / 100000000, 20)
            
            # Breakout bonus
            breakout_bonus = 0
            if is_daily_breakout:
                breakout_bonus = 30
            elif is_15min_breakout:
                breakout_bonus = 20
            
            total_score = momentum_score + volume_score + value_score + breakout_bonus
            return min(total_score, 100)
            
        except:
            return 0
    
    def select_top_stocks(self, gainers_data, losers_data):
        """Select top AI-recommended stocks"""
        gainers = gainers_data.get('data', [])
        losers = losers_data.get('data', [])
        
        # Score and sort gainers
        for stock in gainers:
            stock['ai_score'] = self.calculate_ai_score(stock)
        
        # Score and sort losers
        for stock in losers:
            stock['ai_score'] = self.calculate_ai_score(stock)
        
        # Sort by AI score
        top_gainers = sorted(gainers, key=lambda x: x['ai_score'], reverse=True)[:5]
        top_losers = sorted(losers, key=lambda x: x['ai_score'], reverse=True)[:4]
        
        return top_gainers, top_losers