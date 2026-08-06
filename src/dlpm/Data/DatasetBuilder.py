import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

class TradingDayEstimator:
    """Trading Day Estimator - Predict future trading days based on historical data"""
    
    def __init__(self, price_df):
        self.price_df = price_df.sort_values('trading_date').reset_index(drop=True)
        self.historical_ratios = self._build_historical_ratios()
    
    def _build_historical_ratios(self):
        """Build historical calendar-to-trading day conversion ratios"""
        print("    Building historical trading day conversion ratios...")
        ratios = {}
        
        for calendar_days in [30, 90, 180, 365]:
            ratio_samples = []
            end_idx = int(len(self.price_df) * 0.8)
            
            for i in range(252, end_idx - calendar_days - 10):
                start_date = self.price_df.iloc[i]['trading_date']
                end_date = start_date + pd.Timedelta(days=calendar_days)
                
                mask = ((self.price_df['trading_date'] >= start_date) & 
                       (self.price_df['trading_date'] <= end_date))
                actual_trading_days = mask.sum() - 1
                
                if actual_trading_days > 0:
                    ratio = actual_trading_days / calendar_days
                    ratio_samples.append(ratio)
            
            if ratio_samples:
                avg_ratio = np.mean(ratio_samples)
                std_ratio = np.std(ratio_samples)
                ratios[calendar_days] = {
                    'mean': avg_ratio,
                    'std': std_ratio,
                    'samples': len(ratio_samples)
                }
                print(f"      {calendar_days} days: avg ratio={avg_ratio:.3f}±{std_ratio:.3f}")
        
        return ratios
    
    def estimate_trading_days(self, calendar_days, as_of_date):
        """Estimate trading days for specified calendar days"""
        if calendar_days not in self.historical_ratios:
            return int(calendar_days * 0.7)
        
        ratio_info = self.historical_ratios[calendar_days]
        expected_trading_days = int(calendar_days * ratio_info['mean'])
        return max(1, expected_trading_days)

class DatasetProcessor:
    """
    All-in-One Dataset Processor
    
    One-click generation of training and validation datasets from raw market data
    Supports both China and US markets with unified English column format
    """
    
    def __init__(self, 
                 periods=[30, 90, 180, 365], 
                 vol_lookback=20, 
                 cutoff_date_str='2022-01-01',
                 market: str = "usa",
                 verbose=True):
        """
        Initialize Dataset Processor
        
        Parameters:
        -----------
        periods : list, default=[30, 90, 180, 365]
            Contract periods in days
        vol_lookback : int, default=20
            Volatility lookback period in trading days
        cutoff_date : str, default='2022-01-01'
            Training/validation split date (YYYY-MM-DD)
        verbose : bool, default=True
            Whether to print detailed progress information
        """
        self.periods = periods
        self.vol_lookback = vol_lookback
        self.cutoff_date = pd.to_datetime(cutoff_date_str)
        self.verbose = verbose
        self.estimator = None
        
        #if market == "usa":
            #self.cutoff_date = self.cutoff_date.tz_localize("America/New_York")

        if self.verbose:
            print("🚀 DatasetProcessor Initialized")
            print(f"   Periods: {self.periods} days")
            print(f"   Volatility lookback: {self.vol_lookback} days")
            print(f"   Train/Val split: {cutoff_date_str}")
    
    def process_all(self, df_stock, rates_dict):
        """
        One-click processing: Generate both training and validation datasets
        
        Parameters:
        -----------
        df_stock : pd.DataFrame
            Stock data with English columns: ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'country']
        rates_dict : dict
            Interest rates dictionary: {period_days: DataFrame}
            Each DataFrame has columns: ['date', 'rate', 'ticker', 'country']
            
        Returns:
        --------
        tuple: (train_df, val_df, estimator)
            train_df : pd.DataFrame - Training dataset
            val_df : pd.DataFrame - Validation dataset  
            estimator : TradingDayEstimator - Trading day estimator
        """
        if self.verbose:
            print("\n" + "="*60)
            print("🔄 ONE-CLICK DATASET PROCESSING")
            print("="*60)
            print(f"📊 Input Data Summary:")
            print(f"   Stock: {df_stock['ticker'].iloc[0]} ({df_stock['country'].iloc[0]})")
            print(f"   Records: {len(df_stock):,}")
            print(f"   Date range: {df_stock['date'].min()} to {df_stock['date'].max()}")
            print(f"   Rate periods: {list(rates_dict.keys())} days")
        
        # Step 1: Data validation and preprocessing
        self._validate_input_data(df_stock, rates_dict)
        df_processed, rates_processed = self._preprocess_data(df_stock, rates_dict)
        
        # Step 2: Build trading day estimator
        if self.verbose:
            print("\n🔧 Building Trading Day Estimator...")
        df_processed['trading_date'] = df_processed['trading_date'].dt.tz_localize(None)
        if hasattr(self.cutoff_date, 'tzinfo') and self.cutoff_date.tzinfo is not None:
           self.cutoff_date = self.cutoff_date.replace(tzinfo=None)
        self.estimator = TradingDayEstimator(df_processed[df_processed['trading_date'] <= self.cutoff_date])
        
        # Step 3: Generate training dataset
        if self.verbose:
            print("\n📈 Generating Training Dataset...")
        train_df = self._generate_training_samples(df_processed, rates_processed)
        
        # Step 4: Generate validation dataset
        if self.verbose:
            print("\n📊 Generating Validation Dataset...")
        val_df = self._generate_validation_samples(df_processed, rates_processed)
        
        # Step 5: Quality summary
        self._print_quality_summary(train_df, val_df)
        
        if self.verbose:
            print("✅ ONE-CLICK PROCESSING COMPLETED!")
            print("="*60)
        
        return train_df, val_df, self.estimator
    
    def _validate_input_data(self, df_stock, rates_dict):
        """Validate input data format and content"""
        # Validate stock data
        required_stock_cols = ['date', 'close', 'ticker', 'country']
        missing_cols = set(required_stock_cols) - set(df_stock.columns)
        if missing_cols:
            raise ValueError(f"Stock data missing required columns: {missing_cols}")
        
        if len(df_stock) == 0:
            raise ValueError("Stock data is empty")
        
        # Validate rates data
        if not rates_dict:
            raise ValueError("Rates dictionary is empty")
        
        for period, rate_df in rates_dict.items():
            if rate_df is None or len(rate_df) == 0:
                if self.verbose:
                    print(f"   ⚠️ Warning: {period}-day rate data is empty")
                continue
            
            required_rate_cols = ['date', 'rate']
            missing_rate_cols = set(required_rate_cols) - set(rate_df.columns)
            if missing_rate_cols:
                raise ValueError(f"{period}-day rate data missing columns: {missing_rate_cols}")
    
    def _preprocess_data(self, df_stock, rates_dict):
        """Preprocess data to internal format"""
        # Process stock data
        df_processed = df_stock.copy()
        df_processed = df_processed.rename(columns={'date': 'trading_date', 'close': 'close_price'})
        df_processed['trading_date'] = pd.to_datetime(df_processed['trading_date'])
        df_processed = df_processed.sort_values('trading_date').reset_index(drop=True)
        
        # Process rates data
        rates_processed = {}
        for period_days in self.periods:
            if period_days not in rates_dict or rates_dict[period_days] is None:
                if self.verbose:
                    print(f"   ⚠️ {period_days}-day rate not available, skipping")
                continue
            
            rate_df = rates_dict[period_days].copy()
            rate_df = rate_df.rename(columns={'date': 'trading_date'})
            rate_df['trading_date'] = pd.to_datetime(rate_df['trading_date'])
            rate_df = rate_df.dropna(subset=['trading_date', 'rate'])
            rate_df = rate_df.sort_values('trading_date').drop_duplicates(
                subset=['trading_date'], keep='last').reset_index(drop=True)
            
            if len(rate_df) > 0:
                rate_series = pd.Series(
                    data=rate_df['rate'].values,
                    index=rate_df['trading_date'].values
                ).sort_index()
                rates_processed[period_days] = rate_series
                
                if self.verbose:
                    avg_rate = rate_series.mean() * 100
                    print(f"   ✅ {period_days}-day rate: {len(rate_df)} records, avg {avg_rate:.2f}%")
        
        return df_processed, rates_processed
    
    def _generate_training_samples(self, df_processed, rates_processed):
        """Generate training samples"""
        # Use only training period data
        df_processed['trading_date'] = df_processed['trading_date'].dt.tz_localize(None)
        if hasattr(self.cutoff_date, 'tzinfo') and self.cutoff_date.tzinfo is not None:
           self.cutoff_date = self.cutoff_date.replace(tzinfo=None)

        df_train = df_processed[df_processed['trading_date'] <= self.cutoff_date].copy()
        
        all_samples = []
        
        for calendar_days in self.periods:
            if calendar_days not in rates_processed:
                continue
            
            rate_series = rates_processed[calendar_days]
            samples_count = 0
            
            if self.verbose:
                print(f"   📈 Processing {calendar_days}-day contracts...")
            
            # Generate samples with progress bar
            iterator = range(self.vol_lookback, len(df_train) - calendar_days - 10)
            if self.verbose:
                iterator = tqdm(iterator, desc=f"{calendar_days}d train", leave=False)
            
            for start_idx in iterator:
                start_date = df_train.iloc[start_idx]['trading_date']
                start_price = df_train.iloc[start_idx]['close_price']
                
                # Predict trading days (no future information)
                expected_trading_days = self.estimator.estimate_trading_days(calendar_days, start_date)
                
                # Get actual future price path
                end_date = start_date + pd.Timedelta(days=calendar_days)
                price_mask = ((df_train['trading_date'] >= start_date) & 
                             (df_train['trading_date'] <= end_date))
                future_prices = df_train[price_mask]['close_price'].tolist()
                
                if len(future_prices) < 2:
                    continue
                
                # Get risk-free rate
                try:
                    risk_free_rate = rate_series.asof(start_date)
                    if pd.isna(risk_free_rate):
                        continue
                except:
                    continue

                # Calculate historical volatility
                past_prices = df_train.iloc[start_idx - self.vol_lookback : start_idx]['close_price']
                log_returns = np.log(past_prices / past_prices.shift(1)).dropna()
                if len(log_returns) < 2: 
                    continue
                volatility = log_returns.std() * np.sqrt(252)

                all_samples.append({
                    "contract_calendar_days": calendar_days,
                    "expected_trading_days": expected_trading_days,
                    "actual_trading_days": len(future_prices) - 1,
                    "start_date": start_date,
                    "start_price": start_price,
                    "price_series": future_prices,
                    "volatility": volatility,
                    "risk_free_rate": risk_free_rate,
                    "data_type": "train",
                    "ticker": df_processed['ticker'].iloc[0],
                    "country": df_processed['country'].iloc[0]
                })
                samples_count += 1

            if self.verbose:
                print(f"      ✅ Generated {samples_count:,} samples")

        return pd.DataFrame(all_samples)
    
    def _generate_validation_samples(self, df_processed, rates_processed):
        """Generate validation samples"""
        # Find validation start index
        val_start_idx = 0
        df_processed['trading_date'] = df_processed['trading_date'].dt.tz_localize(None)
        if hasattr(self.cutoff_date, 'tzinfo') and self.cutoff_date.tzinfo is not None:
           self.cutoff_date = self.cutoff_date.replace(tzinfo=None)
        val_data = df_processed[df_processed['trading_date'] > self.cutoff_date]
        if len(val_data) > 0:
            val_start_date = val_data['trading_date'].min()
            val_start_idx = df_processed[df_processed['trading_date'] >= val_start_date].index[0]
        
        all_samples = []
        
        for calendar_days in self.periods:
            if calendar_days not in rates_processed:
                continue
            
            rate_series = rates_processed[calendar_days]
            samples_count = 0
            
            if self.verbose:
                print(f"   📊 Processing {calendar_days}-day contracts...")
            
            # Generate samples with progress bar
            iterator = range(max(self.vol_lookback, val_start_idx), 
                           len(df_processed) - calendar_days - 10)
            if self.verbose:
                iterator = tqdm(iterator, desc=f"{calendar_days}d val", leave=False)
            
            for start_idx in iterator:
                start_date = df_processed.iloc[start_idx]['trading_date']
                start_price = df_processed.iloc[start_idx]['close_price']
                
                # Calculate actual future trading day path
                end_date = start_date + pd.Timedelta(days=calendar_days)
                price_mask = ((df_processed['trading_date'] >= start_date) & 
                             (df_processed['trading_date'] <= end_date))
                future_prices = df_processed[price_mask]['close_price'].tolist()
                
                if len(future_prices) < 2:
                    continue
                
                actual_trading_days = len(future_prices) - 1
                
                # Get risk-free rate
                try:
                    risk_free_rate = rate_series.asof(start_date)
                    if pd.isna(risk_free_rate):
                        continue
                except:
                    continue

                # Calculate historical volatility
                past_prices = df_processed.iloc[start_idx - self.vol_lookback : start_idx]['close_price']
                log_returns = np.log(past_prices / past_prices.shift(1)).dropna()
                if len(log_returns) < 2: 
                    continue
                volatility = log_returns.std() * np.sqrt(252)

                all_samples.append({
                    "contract_calendar_days": calendar_days,
                    "actual_trading_days": actual_trading_days,
                    "start_date": start_date,
                    "end_date": end_date,
                    "start_price": start_price,
                    "end_price": future_prices[-1],
                    "price_series": future_prices,
                    "volatility": volatility,
                    "risk_free_rate": risk_free_rate,
                    "data_type": "validation",
                    "ticker": df_processed['ticker'].iloc[0],
                    "country": df_processed['country'].iloc[0]
                })
                samples_count += 1

            if self.verbose:
                print(f"      ✅ Generated {samples_count:,} samples")

        return pd.DataFrame(all_samples)
    
    def _print_quality_summary(self, train_df, val_df):
        """Print dataset quality summary"""
        if not self.verbose:
            return
        
        print("\n📋 DATASET QUALITY SUMMARY")
        print("="*50)
        
        # Training dataset summary
        if len(train_df) > 0:
            print("📈 Training Dataset:")
            print(f"   Total samples: {len(train_df):,}")
            print(f"   Period distribution:")
            for period in sorted(train_df['contract_calendar_days'].unique()):
                count = len(train_df[train_df['contract_calendar_days'] == period])
                print(f"     {period} days: {count:,} samples")
            
            print(f"   Time range: {train_df['start_date'].min().date()} to {train_df['start_date'].max().date()}")
            print(f"   Volatility: {train_df['volatility'].mean():.3f} ± {train_df['volatility'].std():.3f}")
            print(f"   Risk-free rate: {train_df['risk_free_rate'].mean()*100:.2f}% ± {train_df['risk_free_rate'].std()*100:.2f}%")
        else:
            print("❌ Training dataset is empty")
        
        # Validation dataset summary
        if len(val_df) > 0:
            print("\n📊 Validation Dataset:")
            print(f"   Total samples: {len(val_df):,}")
            print(f"   Period distribution:")
            for period in sorted(val_df['contract_calendar_days'].unique()):
                count = len(val_df[val_df['contract_calendar_days'] == period])
                print(f"     {period} days: {count:,} samples")
            
            print(f"   Time range: {val_df['start_date'].min().date()} to {val_df['start_date'].max().date()}")
            print(f"   Volatility: {val_df['volatility'].mean():.3f} ± {val_df['volatility'].std():.3f}")
            print(f"   Risk-free rate: {val_df['risk_free_rate'].mean()*100:.2f}% ± {val_df['risk_free_rate'].std()*100:.2f}%")
        else:
            print("❌ Validation dataset is empty")
        
        print("="*50)

class QuickDatasetBuilder:
    """
    Quick Dataset Builder - Even simpler interface
    
    For users who want maximum convenience with minimal configuration
    """
    
    @staticmethod
    def build_datasets(df_stock, rates_dict, 
                      periods=None, 
                      cutoff_date='2022-01-01',
                      vol_lookback=20):
        """
        Build training and validation datasets with one function call
        
        Parameters:
        -----------
        df_stock : pd.DataFrame
            Stock data from DataProvider
        rates_dict : dict
            Rates data from DataProvider
        periods : list, optional
            Contract periods, defaults to [30, 90, 180, 365]
        cutoff_date : str, optional
            Train/val split date, defaults to '2022-01-01'
        vol_lookback : int, optional
            Volatility lookback period, defaults to 20
            
        Returns:
        --------
        tuple: (train_df, val_df, processor)
            Ready-to-use training and validation datasets
        """
        if periods is None:
            periods = [30, 90, 180, 365]
        
        processor = DatasetProcessor(
            periods=periods,
            vol_lookback=vol_lookback,
            cutoff_date=cutoff_date,
            verbose=True
        )
        
        return processor.process_all(df_stock, rates_dict)
