import os
import glob
import numpy as np
import pandas as pd
from numba import jit
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from datetime import timedelta
from joblib import Parallel, delayed
import gc
import shutil
from tqdm import tqdm

# 设置绘图风格
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. 配置
# -----------------------------------------------------------------------------
class Config:
    # --- 路径配置 ---
    ROOT_DIR = './data'
    PATH_PLOTS = './plots2'
    
    # 基础数据路径
    PATH_HF_BAR = os.path.join(ROOT_DIR, 'bar_1m_2016_2018') 
    PATH_DAILY_QUOTE = os.path.join(ROOT_DIR, '日行情') 
    PATH_TRADING_DAYS = os.path.join(ROOT_DIR, '交易日期', 'tradingdays.csv')
    PATH_INDUSTRY = os.path.join(ROOT_DIR, '股票行业分类', 'industry.csv')
    PATH_CAP = os.path.join(ROOT_DIR, '股票市值', 'stock_cap.parquet')
    
    # 指数成分
    PATH_INDEX_WT_300 = os.path.join(ROOT_DIR, '指数权重', 'index_wt_000300.parquet')
    PATH_INDEX_WT_500 = os.path.join(ROOT_DIR, '指数权重', 'index_wt_000905.parquet')
    
    # --- 运算配置 ---
    N_JOBS = 1            # 顺序处理更高效
    SMART_MONEY_WINDOW = 10 
    
    # --- 交易参数 ---
    TOP_K = 50            # 持仓数量
    COST_RATE_TOTAL = 0.003  # 双边千三
    MAX_GAP_DAYS = 20     # 超过20天无数据视为断层，清空历史

    TEST_PERIODS = {
        'Weekly': 5,
        'BiWeekly': 10,
        'Monthly': 20
    }

# -----------------------------------------------------------------------------
# 2. 核心计算函数
# -----------------------------------------------------------------------------
@jit(nopython=True, cache=True)
def calc_daily_smart_money(open_arr, close_arr, vol_arr, amt_arr):
    n = len(close_arr)
    if n == 0: return np.nan
    
    # 1. 计算收益率和 S 指标
    s_metric = np.zeros(n)
    for i in range(n):

        # 在循环内部，vol_arr 为 0 的处理必须放在前面，否则会导致除以零的错误
        if vol_arr[i] < 1e-6:
            s_metric[i] = -1.0 # 确保排在最后
            continue

        # 收益率计算保持原样
        if i == 0:
            ret = close_arr[i] / open_arr[i] - 1.0 if open_arr[i] > 1e-6 else 0.0
        else:
            ret = close_arr[i] / close_arr[i-1] - 1.0 if close_arr[i-1] > 1e-6 else 0.0
            
        safe_vol = vol_arr[i] if vol_arr[i] > 1e-9 else 0.0
        s_metric[i] = np.abs(ret) / (np.power(safe_vol, 0.25) + 1e-9)
            
    # 2. 排序
    sorted_idx = np.argsort(s_metric)[::-1]
    total_vol = np.sum(vol_arr)
    if total_vol < 1e-6: return np.nan
    
    # 3. 截取前 20% (含插值)
    target_vol = total_vol * 0.20
    current_vol = 0.0
    smart_amt = 0.0
    smart_vol = 0.0 # 理论上应严格等于 target_vol
    
    for idx in sorted_idx:
        v = vol_arr[idx]
        if v <= 0: continue
        
        # 检查是否加上这个 bar 会超过 20%
        if current_vol + v > target_vol:
            remain_vol = target_vol - current_vol
            ratio = remain_vol / v
            
            smart_vol += remain_vol
            smart_amt += amt_arr[idx] * ratio
            current_vol += remain_vol # 达到 target，结束
            break
        else:
            smart_vol += v
            smart_amt += amt_arr[idx]
            current_vol += v
            
    if smart_vol < 1e-6: return np.nan
    
    # 4. 计算 VWAP 比例
    vwap_smart = smart_amt / smart_vol
    vwap_all = np.sum(amt_arr) / total_vol
    
    if vwap_all < 1e-6: return np.nan
    
    return vwap_smart / vwap_all

@jit(nopython=True, cache=True)
def calc_linear_slope(y_arr):
    """
    计算时间序列 y 对时间 t (0, 1, 2...) 的线性回归斜率
    用于 CPV Trend 计算
    """
    n = len(y_arr)
    if n < 3: return 0.0
    
    x_sum = 0.0
    y_sum = 0.0
    xy_sum = 0.0
    xx_sum = 0.0
    
    count = 0
    for i in range(n):
        v = y_arr[i]
        if np.isnan(v): continue # 跳过 NaN
        
        x = float(i)
        x_sum += x
        y_sum += v
        xy_sum += x * v
        xx_sum += x * x
        count += 1
        
    if count < 3: return 0.0
    
    x_mean = x_sum / count
    y_mean = y_sum / count
    
    numerator = xy_sum - count * x_mean * y_mean
    denominator = xx_sum - count * x_mean * x_mean
    
    if np.abs(denominator) < 1e-9:
        return 0.0
    return numerator / denominator

@jit(nopython=True, cache=True)
def calc_subset_correlation(x_arr, y_arr, mode):
    """
    计算特定象限子序列的相关系数
    mode 1: x>0, y>0 (Up-Up)
    mode 2: x<0, y<0 (Down-Down)
    """
    n = len(x_arr)
    if n < 2: return 0.0
    
    sum_x, sum_y = 0.0, 0.0
    sum_xx, sum_yy = 0.0, 0.0
    sum_xy = 0.0
    count = 0
    
    for i in range(n):
        vx = x_arr[i]
        vy = y_arr[i]
        
        # 筛选逻辑
        valid = False
        if mode == 1:
            if vx > 1e-9 and vy > 1e-9: valid = True
        elif mode == 2:
            if vx < -1e-9 and vy < -1e-9: valid = True
            
        if valid:
            sum_x += vx
            sum_y += vy
            sum_xx += vx * vx
            sum_yy += vy * vy
            sum_xy += vx * vy
            count += 1
            
    if count < 3: return 0.0 # 样本过少无法计算相关系数
    
    mean_x = sum_x / count
    mean_y = sum_y / count
    
    cov = sum_xy - count * mean_x * mean_y
    var_x = sum_xx - count * mean_x * mean_x
    var_y = sum_yy - count * mean_y * mean_y

    # 防止精度误差导致 var < 0
    if var_x < 0: var_x = 0
    if var_y < 0: var_y = 0
    
    if var_x > 1e-12 and var_y > 1e-12:
        return cov / np.sqrt(var_x * var_y)
    return 0.0

@jit(nopython=True, cache=True)
def calc_daily_metrics_intraday(open_arr, close_arr, high_arr, low_arr, vol_arr):
    """
    Level 1: 计算基于分钟数据的当日基础指标 (Updated: Skewness & Amplitude)
    Returns: [SM, CPV, CDPDP, High, Low, Close, Ret, Skew, TailRet, Amp] -> Size 10
    """
    n = len(close_arr)
    if n < 30: return np.array([np.nan] * 10)
    
    # --- 1. Smart Money ---
    s_metric = np.zeros(n)
    valid_sm_count = 0
    for i in range(1, n):
        v = vol_arr[i]
        if v < 1e-6:
            s_metric[i] = -1.0 
            continue
        if close_arr[i-1] > 1e-6:
            r = np.abs(close_arr[i] / close_arr[i-1] - 1.0)
            s_metric[i] = r / (np.power(v, 0.25))
            valid_sm_count += 1
        else:
            s_metric[i] = -1.0
            
    sm_factor = np.nan
    if valid_sm_count > 10:
        valid_indices = np.where(s_metric >= 0)[0]
        if len(valid_indices) > 0:
            s_valid = s_metric[valid_indices]
            sorted_args = np.argsort(s_valid)[::-1]
            sorted_real_idx = valid_indices[sorted_args]
            total_vol = np.sum(vol_arr)
            if total_vol > 1e-6:
                target_vol = total_vol * 0.20
                curr_vol = 0.0
                smart_amt = 0.0
                smart_vol = 0.0
                avg_price = close_arr 
                amt_arr = avg_price * vol_arr
                for idx in sorted_real_idx:
                    v = vol_arr[idx]
                    if curr_vol + v > target_vol:
                        remain = target_vol - curr_vol
                        ratio = remain / v
                        smart_vol += remain
                        smart_amt += amt_arr[idx] * ratio
                        break
                    else:
                        smart_vol += v
                        smart_amt += amt_arr[idx]
                        curr_vol += v
                if smart_vol > 1e-6:
                    vwap_smart = smart_amt / smart_vol
                    vwap_all = np.sum(amt_arr) / total_vol
                    if vwap_all > 1e-6:
                        sm_factor = vwap_smart / vwap_all

    # --- 2. CPV Component ---
    mean_c = np.mean(close_arr)
    mean_v = np.mean(vol_arr)
    num = 0.0
    den_c = 0.0
    den_v = 0.0
    for i in range(n):
        dc = close_arr[i] - mean_c
        dv = vol_arr[i] - mean_v
        num += dc * dv
        den_c += dc**2
        den_v += dv**2
    corr_pv = 0.0
    if den_c > 1e-9 and den_v > 1e-9:
        corr_pv = num / np.sqrt(den_c * den_v)
        
    # --- 3. CDPDP ---
    diffs = close_arr[1:] - close_arr[:-1]
    if len(diffs) > 10:
        x_seq = diffs[:-1]
        y_seq = diffs[1:]
        corr_up_up = calc_subset_correlation(x_seq, y_seq, 1)
        corr_down_down = calc_subset_correlation(x_seq, y_seq, 2)
        cdpdp_raw = corr_up_up + corr_down_down
    else:
        cdpdp_raw = np.nan
    
    # --- 4. Daily Basic Stats (保持不变) ---
    day_high = np.max(high_arr)
    day_low = np.min(low_arr)
    day_close = close_arr[-1]
    if open_arr[0] > 1e-6:
        day_ret_intraday = day_close / open_arr[0] - 1.0 
    else:
        day_ret_intraday = 0.0

    # ==========================================
    # --- 5. New High-Freq Factors ---
    # ==========================================

    # A. Skewness (偏度) - 替换了 Vol
    # 计算公式: E[(x-u)^3] / std^3
    val_skew = np.nan
    if n > 10:
        # 1. 计算分钟收益率
        rets = np.zeros(n-1)
        count = 0
        sum_r = 0.0
        for i in range(1, n):
            if close_arr[i-1] > 1e-6:
                r = close_arr[i] / close_arr[i-1] - 1.0
                rets[i-1] = r
                sum_r += r
                count += 1
        
        if count > 10:
            mean_r = sum_r / count
            sum_sq = 0.0
            sum_cu = 0.0
            
            for i in range(n-1):
                if rets[i] == 0 and i > count: continue # Skip padding if any
                diff = rets[i] - mean_r
                sum_sq += diff * diff
                sum_cu += diff * diff * diff
            
            var_r = sum_sq / count
            if var_r > 1e-12:
                std_r = np.sqrt(var_r)
                m3 = sum_cu / count
                val_skew = m3 / (std_r * std_r * std_r)
            else:
                val_skew = 0.0

    # B. Tail Return (尾盘收益) - 保留
    val_tail = np.nan
    lookback = 30
    if n >= lookback:
        start_price = close_arr[-lookback]
        end_price = close_arr[-1]
        if start_price > 1e-6:
            val_tail = end_price / start_price - 1.0
    else:
        start_idx = int(n * 0.75)
        start_price = close_arr[start_idx]
        end_price = close_arr[-1]
        if start_price > 1e-6:
            val_tail = end_price / start_price - 1.0

    # C. Amplitude (振幅) - 替换了 Efficiency
    # (High - Low) / Open
    val_amp = np.nan
    if open_arr[0] > 1e-6:
        val_amp = (day_high - day_low) / open_arr[0]

    # Return size: 10
    return np.array([
        sm_factor,      # 0
        corr_pv,        # 1
        cdpdp_raw,      # 2
        day_high,       # 3
        day_low,        # 4
        day_close,      # 5
        day_ret_intraday, # 6
        val_skew,       # 7 (New: Skew)
        val_tail,       # 8 (Old: Tail)
        val_amp         # 9 (New: Amp)
    ])

# -----------------------------------------------------------------------------
# 3. 数据加载器
# -----------------------------------------------------------------------------
class DataLoader:
    @staticmethod
    def load_universe(index_path):
        # 保持原样，无需修改
        print(f">>> [Loader] 加载股票池: {index_path} ...")
        df = pd.read_parquet(index_path)
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'symbol']].drop_duplicates()

    @staticmethod
    def load_daily_quote():
        # 保持原样，无需修改
        print(">>> [Loader] 加载日行情 ...")
        files = glob.glob(os.path.join(Config.PATH_DAILY_QUOTE, "*.csv"))
        dfs = []
        cols = [
            'SecuCode', 'TradingDay', 'OpenPrice', 'ClosePrice', 'PTFlag',
            'TradingVolumes', 'TurnoverValue', 'NonRestrictedShares', 'AFloats', 
        ] 
        for f in tqdm(files, desc="[Loader] Reading CSVs"):
            try:
                temp = pd.read_csv(f, usecols=lambda c: c in cols, dtype={'SecuCode': str})
                dfs.append(temp)
            except ValueError:
                print(f"Warning: File {f} format mismatch, skipping.")
        
        if not dfs:
            raise ValueError("No daily quote files found!")

        df = pd.concat(dfs)
        df['TradingDay'] = pd.to_datetime(df['TradingDay'])
        df['SecuCode'] = df['SecuCode'].str.zfill(6)
        
        df.rename(columns={
            'TradingVolumes': 'Volume', 
            'TurnoverValue': 'Amount'
        }, inplace=True)
        
        df = df.sort_values(['SecuCode', 'TradingDay'])
        return df

    @staticmethod
    def load_industry_cap():
        # 保持原样，无需修改
        print(">>> [Loader] 加载行业与市值 ...")
        df_ind = pd.read_csv(Config.PATH_INDUSTRY, dtype={'SecuCode': str})
        df_ind['InfoPublDate'] = pd.to_datetime(df_ind['InfoPublDate'])
        df_ind['InfoPublDate'] = df_ind['InfoPublDate'] + pd.Timedelta(days=1)
        df_ind['SecuCode'] = df_ind['SecuCode'].str.zfill(6)
        df_ind = df_ind.sort_values('InfoPublDate')
        
        df_cap = pd.read_parquet(Config.PATH_CAP)
        df_cap.rename(columns={'date': 'TradingDay', 'symbol': 'SecuCode', 'tot_cap': 'Cap'}, inplace=True)
        df_cap['TradingDay'] = pd.to_datetime(df_cap['TradingDay'])
        df_cap['SecuCode'] = df_cap['SecuCode'].astype(str).str.zfill(6)
        df_cap['Cap'] = df_cap['Cap'].astype(float)
        df_cap['LnCap'] = np.log(df_cap['Cap'] + 1)
        return df_ind, df_cap

# -----------------------------------------------------------------------------
# 4. 因子引擎
# -----------------------------------------------------------------------------
class FactorEngine:
    def __init__(self, valid_universe_df, cache_dir='./temp_factors'):
        self.daily_files = sorted(glob.glob(os.path.join(Config.PATH_HF_BAR, "*.parquet")))
        self.univ_map = valid_universe_df.groupby('date')['symbol'].apply(set).to_dict()
        self.cache_dir = cache_dir
        self.batch_size = 50000 
        self.buffer = []       
        self.part_idx = 0      
        
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.MAX_HISTORY = 170
        self.history = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
        self.MAX_GAP_DAYS = 30 

    def _flush_buffer(self):
        if not self.buffer: return
        df_chunk = pd.DataFrame(self.buffer)
        cols = df_chunk.select_dtypes(include=['float64']).columns
        df_chunk[cols] = df_chunk[cols].astype('float32')
        save_path = os.path.join(self.cache_dir, f'part_{self.part_idx:04d}.parquet')
        df_chunk.to_parquet(save_path, index=False)
        self.part_idx += 1
        self.buffer = []
        del df_chunk
        gc.collect()

    def run(self):
        print(f">>> [Factor] 开始计算高频与滚动因子 (Raw Value Version)...")
        file_iterator = tqdm(self.daily_files, desc="[Factor] Processing Days")

        for f_path in file_iterator:
            date_str = os.path.basename(f_path).split('.')[0]
            curr_date = pd.Timestamp(date_str)
            
            try:
                df_day = pd.read_parquet(f_path)
                df_day.columns = [c.lower() for c in df_day.columns]
            except: continue
                
            if curr_date not in self.univ_map: continue
            valid_symbols = self.univ_map[curr_date]
            df_day = df_day[df_day['symbol'].isin(valid_symbols)]
            if df_day.empty: continue
            
            g = df_day.groupby('symbol')
            
            for sym, sub_df in g:
                if 'time' in sub_df.columns: sub_vals = sub_df.sort_values('time')
                else: sub_vals = sub_df
                
                daily_mets = calc_daily_metrics_intraday(
                    sub_vals['open'].values, sub_vals['close'].values,
                    sub_vals['high'].values, sub_vals['low'].values,
                    sub_vals['volume'].values
                )
                
                queue = self.history[sym]
                
                if len(queue) > 0:
                    last_date, last_mets = queue[-1]
                    if (curr_date - last_date).days > self.MAX_GAP_DAYS:
                        queue.clear()
                    else:
                        prev_close = last_mets[5] 
                        curr_close = daily_mets[5]
                        if prev_close > 1e-6:
                            true_ret = curr_close / prev_close - 1.0
                            daily_mets[6] = true_ret
                        else:
                            daily_mets[6] = 0.0

                if not np.isnan(daily_mets[5]): 
                    queue.append((curr_date, daily_mets))
                
                n_hist = len(queue)
                if n_hist < 20: continue 
                
                hist_data = np.array([item[1] for item in queue]) 
                
                # --- Factor 1: Smart Money ---
                sm_vals = hist_data[-10:, 0]
                sm_vals = sm_vals[sm_vals > -0.5] 
                val_sm = np.mean(sm_vals) if len(sm_vals) > 0 else np.nan
                
                # --- Factor 2: CPV Mean ---
                val_cpv_mean = np.nan
                if n_hist >= 20: 
                    cpv_slice = hist_data[-20:, 1]
                    mask = ~np.isnan(cpv_slice)
                    if np.sum(mask) > 15:
                        val_cpv_mean = np.mean(cpv_slice[mask]) 

                # --- Factor 3: CDPDP ---
                cdpdp_slice = hist_data[-20:, 2] 
                cdpdp_slice = cdpdp_slice[~np.isnan(cdpdp_slice)]
                val_cdpdp = np.mean(cdpdp_slice) if len(cdpdp_slice) > 0 else np.nan
                
                # --- Factor 4: HF_Skewness ---
                skew_slice = hist_data[-20:, 7]
                skew_slice = skew_slice[~np.isnan(skew_slice)]
                val_skew = np.mean(skew_slice) if len(skew_slice) > 0 else np.nan

                # --- Factor 5: HF_Tail_Reversal ---
                tail_slice = hist_data[-20:, 8]
                tail_slice = tail_slice[~np.isnan(tail_slice)]
                val_tail_rev = np.mean(tail_slice) if len(tail_slice) > 0 else np.nan

                # --- Factor 6: HF_Amplitude  ---
                amp_slice = hist_data[-20:, 9]
                amp_slice = amp_slice[~np.isnan(amp_slice)]
                val_amp = np.mean(amp_slice) if len(amp_slice) > 0 else np.nan

                # 4. 写入 Buffer (注意：这里去掉了所有的 -1.0 乘法)
                self.buffer.append({
                    'TradingDay': curr_date,
                    'SecuCode': str(sym).zfill(6),
                    'HF_SmartMoney': val_sm,
                    'HF_Raw_CPV_Mean': val_cpv_mean,  # Raw
                    'HF_CDPDP': val_cdpdp,            # Raw
                    'HF_Skewness': val_skew,          # Raw
                    'HF_Tail_Reversal': val_tail_rev, # Raw
                    'HF_Amplitude': val_amp           # Raw
                })
            
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
                
            file_iterator.set_postfix(date=date_str)

        self._flush_buffer()
        print(">>> [Factor] 计算完成 (Raw Values)，正在合并...")
        try:
            df_final = pd.read_parquet(self.cache_dir)
            return df_final
        except Exception as e:
            print(f"Error merging parquet files: {e}")
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# 5. 数据处理与特征工程
# -----------------------------------------------------------------------------
def process_features(df_full, temp_dir='./temp_processed_features'):
    print(f">>> [Process] 执行特征工程 (去极值、行业市值中性化、正交化)...")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    df_full = df_full.sort_values(['SecuCode', 'TradingDay'])

    # 1. 预计算控制变量 (用于 CPV 正交化)
    print("    [1/3] 计算风格控制变量 (Reversal, Volatility, Turnover)...")
    df_full['Ctrl_Ret20'] = df_full.groupby('SecuCode')['ClosePrice'].pct_change(20)
    df_full['Ctrl_Vol20'] = df_full.groupby('SecuCode')['ClosePrice'].transform(lambda x: x.pct_change().rolling(20).std())
    
    if 'TurnoverValue' in df_full.columns: df_full['Amount'] = df_full['TurnoverValue']
    if 'TradingVolumes' in df_full.columns: df_full['Volume'] = df_full['TradingVolumes']
        
    if 'NonRestrictedShares' in df_full.columns: df_full['FloatShares'] = df_full['NonRestrictedShares']
    elif 'AFloats' in df_full.columns: df_full['FloatShares'] = df_full['AFloats']

    has_turnover = False
    if 'Volume' in df_full.columns and 'FloatShares' in df_full.columns:
        valid_cap = df_full['FloatShares'] > 1e-6
        df_full['Turnover_Daily'] = np.nan
        df_full.loc[valid_cap, 'Turnover_Daily'] = df_full.loc[valid_cap, 'Volume'] / df_full.loc[valid_cap, 'FloatShares']
        df_full['Turnover_Daily'] = df_full['Turnover_Daily'].clip(0.0, 2.0)
        df_full['Ctrl_Turnover20'] = df_full.groupby('SecuCode')['Turnover_Daily'].transform(lambda x: x.rolling(20, min_periods=5).mean())
        df_full['Ctrl_Turnover20'].fillna(df_full['Turnover_Daily'], inplace=True)
        has_turnover = True

    ctrl_cols = ['Ctrl_Ret20', 'Ctrl_Vol20']
    if has_turnover: ctrl_cols.append('Ctrl_Turnover20')
    df_full[ctrl_cols] = df_full[ctrl_cols].fillna(0)

    # (注意：删除了Label行业中性化的逻辑)
    if 'Industry' not in df_full.columns: df_full['Industry'] = 'Unknown'

    # 2. 特征清洗与正交化
    print(f"    [2/3] 分批特征处理 (去极值 -> 中性化 -> Z-Score)...")
    feature_cols = [c for c in df_full.columns if c.startswith('HF_')]
    cpv_cols = [c for c in feature_cols if 'CPV' in c]

    def _process_day_logic(sub_df):
        sub_df = sub_df.dropna(subset=feature_cols, how='all')
        if len(sub_df) < 20: return pd.DataFrame() 
        for col in feature_cols: sub_df[col] = sub_df[col].fillna(sub_df[col].median()).fillna(0)
        
        X_ind = pd.get_dummies(sub_df['Industry'], prefix='Ind')
        X_base = pd.concat([X_ind, sub_df[['LnCap']]], axis=1).astype(float) if X_ind.shape[1] > 0 else sub_df[['LnCap']].astype(float)
        X_extended = pd.concat([X_base, sub_df[ctrl_cols]], axis=1).astype(float)
        X_base, X_extended = X_base.fillna(0), X_extended.fillna(0)
            
        model = LinearRegression()
        for col in feature_cols:
            y = sub_df[col].values
            # MAD 去极值
            median, mad = np.median(y), np.mean(np.abs(y - np.median(y)))
            y = np.clip(y, median - 5 * mad, median + 5 * mad)
            # 差异化正交化
            target_X = X_extended if col in cpv_cols else X_base
            if len(target_X) > 0:
                model.fit(target_X, y)
                y = y - model.predict(target_X)
            # Z-Score
            std_val = np.std(y)
            sub_df[col] = 0 if std_val == 0 else (y - np.mean(y)) / (std_val + 1e-9)
        return sub_df

    df_full['Month'] = df_full['TradingDay'].dt.to_period('M')
    for month in tqdm(df_full['Month'].unique(), desc="Processing Batches"):
        batch_df = df_full[df_full['Month'] == month]
        processed_list = Parallel(n_jobs=Config.N_JOBS)(delayed(_process_day_logic)(group) for _, group in batch_df.groupby('TradingDay'))
        processed_list = [res for res in processed_list if not res.empty]
        if processed_list:
            batch_result = pd.concat(processed_list)
            # 仅保留需要的列 (去除了 Ret_Hold 等)
            keep_cols = ['TradingDay', 'SecuCode', 'LnCap', 'Industry'] + feature_cols
            batch_result = batch_result[[c for c in keep_cols if c in batch_result.columns]]
            for col in [c for c in batch_result.columns if c not in ['TradingDay', 'SecuCode', 'Industry']]:
                try: batch_result[col] = batch_result[col].astype('float32')
                except: pass
            batch_result.to_parquet(os.path.join(temp_dir, f"batch_{month}.parquet"), index=False)

    print("    [3/3] 合并最终数据集...")
    df_final = pd.read_parquet(temp_dir).sort_values(['TradingDay', 'SecuCode'])
    if 'Industry' in df_final.columns: df_final['Industry'] = df_final['Industry'].astype('category')
    return df_final

# -----------------------------------------------------------------------------
# 6. 因子分析模块 (覆盖原有的 FactorAnalyzer 类)
# -----------------------------------------------------------------------------
class FactorAnalyzer:
    @staticmethod
    def calc_forward_returns(df_daily, periods):
        print(">>> [Analyzer] 计算各频率 Forward Returns (T+1 Open 买入, T+1+N Open 卖出)...")
        df_daily = df_daily.sort_values(['TradingDay', 'SecuCode'])
        
        # 提取 OpenPrice
        df_open = df_daily.pivot(index='TradingDay', columns='SecuCode', values='OpenPrice')
        
        # 【核心修复 1】：把异常的 0 价、极小价强制变成 NaN，防止除以 0 产生 inf
        df_open[df_open <= 1e-6] = np.nan
        
        forward_rets = {}
        for name, days in periods.items():
            buy_price = df_open.shift(-1)
            sell_price = df_open.shift(-1 - days)
            
            # 计算收益率
            ret_n = sell_price / buy_price - 1.0
            
            # 【核心修复 2】：为防万一，把任何漏网的 inf 替换为 NaN
            ret_n = ret_n.replace([np.inf, -np.inf], np.nan)
            
            # 堆叠回长格式
            ret_stack = ret_n.stack().reset_index()
            ret_stack.columns = ['TradingDay', 'SecuCode', f'Ret_{name}']
            forward_rets[name] = ret_stack
            
        return forward_rets

    @staticmethod
    def plot_dual_axis(group_nav, ls_nav, title, metrics_text):
        """绘制双轴图并保存"""
        fig, ax1 = plt.subplots(figsize=(10, 8))
        
        colors = sns.color_palette("RdYlGn", 5) 
        for i in range(5):
            nav_series = group_nav[i]
            if not nav_series.empty:
                start_date = nav_series.index[0] - pd.Timedelta(days=1)
                nav_series = pd.concat([pd.Series([1.0], index=[start_date]), nav_series])
            ax1.plot(nav_series.index, nav_series, color=colors[i], alpha=0.8, linewidth=1.5, label=f'Group {i+1}')
            
        ax1.set_ylabel('Group Net Value (Base=1)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.legend(loc='upper left', title='Groups')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        if not ls_nav.empty:
            start_date = ls_nav.index[0] - pd.Timedelta(days=1)
            ls_nav_plot = pd.concat([pd.Series([1.0], index=[start_date]), ls_nav])
        else:
            ls_nav_plot = ls_nav

        ax2.plot(ls_nav_plot.index, ls_nav_plot, color='black', linestyle='--', linewidth=2.5, label='Long-Short (Net)')
        ax2.fill_between(ls_nav_plot.index, ls_nav_plot, 1.0, where=(ls_nav_plot>=1.0), facecolor='red', alpha=0.1)
        ax2.fill_between(ls_nav_plot.index, ls_nav_plot, 1.0, where=(ls_nav_plot<1.0), facecolor='green', alpha=0.1)
        
        ax2.set_ylabel('L/S Net Value (Base=1)', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc='upper right')

        plt.title(title, fontsize=15)
        plt.tight_layout()

        if not os.path.exists(Config.PATH_PLOTS): os.makedirs(Config.PATH_PLOTS, exist_ok=True)
        clean_title = title.replace(' ', '_').replace('|', '-').replace(':', '').replace('/', '-')
        save_path = os.path.join(Config.PATH_PLOTS, f"{clean_title}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 

    @staticmethod
    def analyze_single_period(df_merged, factor_col, ret_col, period_name, period_days, is_ascending):
        data = df_merged[['TradingDay', 'SecuCode', factor_col, ret_col]].dropna()
        if data.empty: return None
        
        # 1. 每日分组
        try:
            data['Group'] = data.groupby('TradingDay')[factor_col].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
        except: return None
        
        # 2. 截面均值收益计算
        daily_group_ret_N = data.groupby(['TradingDay', 'Group'])[ret_col].mean().unstack()
        
        # 统一方向：让 Group 4 永远代表我们想买入的“多头组”
        if not is_ascending:
            daily_group_ret_N = daily_group_ret_N[daily_group_ret_N.columns[::-1]]
            daily_group_ret_N.columns = range(5)

        ls_ret_N_raw = daily_group_ret_N[4] - daily_group_ret_N[0]
        
        # 3. 降采样 (Resampling)
        resampled_idx = ls_ret_N_raw.index[::period_days] 
        # 【核心修复 3】：填充 NaN 为 0，防止某一日截面缺失导致整条净值曲线变成 NaN
        ls_ret_N_resampled = ls_ret_N_raw.loc[resampled_idx].fillna(0.0)
        
        # 4. IC 计算 (加入年化 ICIR)
        def calc_ic(sub):
            if len(sub) < 10: return pd.Series({'IC': np.nan, 'RankIC': np.nan})
            return pd.Series({
                'IC': sub[factor_col].corr(sub[ret_col], method='pearson'),
                'RankIC': sub[factor_col].corr(sub[ret_col], method='spearman')
            })

        daily_ic_df = data.groupby('TradingDay').apply(calc_ic)
        resampled_ic_df = daily_ic_df.loc[resampled_idx]
        
        ic_mean = resampled_ic_df['IC'].mean()
        ic_std = resampled_ic_df['IC'].std()
        rank_ic_mean = resampled_ic_df['RankIC'].mean()
        
        count_per_year = 250.0 / period_days
        ann_icir = (ic_mean / ic_std) * np.sqrt(count_per_year) if ic_std > 1e-6 else 0.0
        
        # 5. 换手率计算
        target_raw_group_idx = 4 if is_ascending else 0
        daily_long_stocks = data[data['Group'] == target_raw_group_idx].groupby('TradingDay')['SecuCode'].apply(set)
        
        turnover_vals = []
        valid_dates = sorted(daily_long_stocks.index)
        rebal_dates = [d for d in resampled_idx if d in valid_dates]
        
        for i in range(1, len(rebal_dates)):
            curr_set, prev_set = daily_long_stocks[rebal_dates[i]], daily_long_stocks[rebal_dates[i-1]]
            if len(curr_set) > 0:
                turnover_vals.append(1.0 - (len(curr_set.intersection(prev_set)) / len(curr_set)))
        
        avg_turnover = np.mean(turnover_vals) if turnover_vals else 0.0

        # 6. 多空 (L/S) 指标 (【修复】: 按照实际平均换手率扣除手续费，且多空包含两头仓位的换手)
        cost_per_period = Config.COST_RATE_TOTAL 
        # 多头和空头都需要换仓，因此粗略估计成本为: 周期收益 - (双边费率 * 单腿换手率 * 2)
        ls_ret_N_net = ls_ret_N_resampled - (cost_per_period * avg_turnover * 2)
        
        ann_ls_ret = (1 + ls_ret_N_net.mean()) ** count_per_year - 1
        ann_ls_vol = ls_ret_N_net.std() * np.sqrt(count_per_year)
        ls_sharpe = ann_ls_ret / ann_ls_vol if ann_ls_vol > 1e-6 else 0
        max_dd = ((1 + ls_ret_N_net).cumprod() / (1 + ls_ret_N_net).cumprod().cummax() - 1).min()

        # 7. 纯多头 (Long Only) 指标 (【修复】: 多头仅承担单腿的真实换手成本)
        long_ret_N_raw = daily_group_ret_N[4].loc[resampled_idx].fillna(0.0)
        long_ret_N_net = long_ret_N_raw - (cost_per_period * avg_turnover)
        
        ann_long_ret = (1 + long_ret_N_net.mean()) ** count_per_year - 1
        ann_long_vol = long_ret_N_net.std() * np.sqrt(count_per_year)
        long_sharpe = ann_long_ret / ann_long_vol if ann_long_vol > 1e-6 else 0

        # 构建文本与日志
        metrics_text = (
            f"IC Mean:     {ic_mean:.4f}\n"
            f"RankIC Mean: {rank_ic_mean:.4f}\n"
            f"Ann. ICIR:   {ann_icir:.2f}\n"
            f"L/S Ret/Vol: {ls_sharpe:.2f}\n"
            f"Long Sharpe: {long_sharpe:.2f}\n"
            f"Turnover:    {avg_turnover:.2%}"
        )

        cli_msg = (f"[{period_name: <8}] "
                   f"IC:{ic_mean:.3f} | ICIR:{ann_icir:.2f} | "
                   f"L/S Ret:{ann_ls_ret:.1%} (SR:{ls_sharpe:.2f}) | "
                   f"Long Ret:{ann_long_ret:.1%} (SR:{long_sharpe:.2f}) | "
                   f"MaxDD:{max_dd:.1%} | Turn:{avg_turnover:.2f}")

        # 准备绘图曲线 (还原到日频以平滑显示)
        group_nav = (1 + daily_group_ret_N / period_days).cumprod()
        ls_nav = (1 + (ls_ret_N_raw - cost_per_period) / period_days).cumprod()
        
        return {
            'group_nav': group_nav,
            'ls_nav': ls_nav,
            'metrics_str': metrics_text,
            'cli_msg': cli_msg 
        }

def plot_factor_correlation(df, method='spearman', title='Factor Correlation Matrix'):
    """
    计算因子相关性并保存热力图到文件夹。
    """
    print(f"--- Plotting Correlation Matrix ({method}) ---")
    
    # 1. 筛选因子列
    factor_cols = [c for c in df.columns if c.startswith('HF_')]
    if not factor_cols:
        print("No columns starting with 'HF_' found.")
        return

    # 2. 计算相关性
    corr = df[factor_cols].corr(method=method)

    # 3. 绘图
    plt.figure(figsize=(12, 10))
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    
    sns.heatmap(
        corr, 
        mask=mask,            
        cmap='coolwarm',      
        vmax=1.0, vmin=-1.0,  
        center=0,             
        annot=True,           
        fmt='.2f',            
        square=True,          
        linewidths=.5,        
        cbar_kws={"shrink": .5} 
    )

    plt.title(f"{title} ({method.capitalize()})", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # --- 保存图片而不是显示 ---
    # 1. 确保文件夹存在
    if not os.path.exists(Config.PATH_PLOTS):
        os.makedirs(Config.PATH_PLOTS, exist_ok=True)
    
    # 2. 构造文件名 (清理非法字符)
    clean_title = title.replace(' ', '_').replace('|', '-').replace('/', '-')
    filename = f"{clean_title}_{method}.png"
    save_path = os.path.join(Config.PATH_PLOTS, filename)
    
    # 3. 保存并关闭
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # 关闭画布，防止内存泄露
    print(f">>> Plot saved to: {save_path}")

# -----------------------------------------------------------------------------
# 集成到 Pipeline (覆盖原函数)
# -----------------------------------------------------------------------------
def run_factor_analysis_step(df_final, df_daily_quote):
    print(f"\n{'='*60}")
    print(f"Starting Pure Factor Analysis (Replicating Report Metrics)")
    print(f"{'='*60}")
    
    df_analysis = df_final.copy()
    
    # 1. 计算各频率的 Forward Return
    forward_rets_dict = FactorAnalyzer.calc_forward_returns(df_daily_quote, Config.TEST_PERIODS)
    
    # 2. 定义因子与方向 (移除 Pred_Score)
    factor_directions = {
        'HF_SmartMoney': False,       
        'HF_Raw_CPV_Mean': False,     
        'HF_CDPDP': False,            
        'HF_Skewness': False,         
        'HF_Tail_Reversal': False,    
        'HF_Amplitude': False
    }

    # 3. 循环分析
    for factor, is_ascending in factor_directions.items():
        if factor not in df_analysis.columns: continue
        
        direction_str = "Ascending" if is_ascending else "Descending"
        print(f"\n>>> Analyzing Factor: {factor} [{direction_str}]")
        
        for period_name, period_days in Config.TEST_PERIODS.items():
            ret_df = forward_rets_dict[period_name]
            
            # Merge factor and specific period return
            df_merged = pd.merge(
                df_analysis[['TradingDay', 'SecuCode', factor]], 
                ret_df, 
                on=['TradingDay', 'SecuCode'], 
                how='inner'
            )
            
            col_ret_name = f'Ret_{period_name}'
            
            res = FactorAnalyzer.analyze_single_period(
                df_merged, factor, col_ret_name, period_name, period_days, is_ascending
            )
            
            if res:
                title = f"Factor: {factor} | Freq: {period_name} | {direction_str}"
                FactorAnalyzer.plot_dual_axis(res['group_nav'], res['ls_nav'], title, res['metrics_str'])
                print(f"  {res['cli_msg']}")

    # 4. 相关性图表
    plot_factor_correlation(df_analysis, method='spearman')

# -----------------------------------------------------------------------------
# Main Pipeline 
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main Pipeline (覆盖原函数)
# -----------------------------------------------------------------------------
def run_pipeline(index_path, universe_name):
    print(f"\n{'='*60}")
    print(f"Running Pure Factor Pipeline for: {universe_name} (Report Replication)")
    print(f"{'='*60}")
    
    # 1. 数据加载
    try:
        df_univ = DataLoader.load_universe(index_path)
        df_daily = DataLoader.load_daily_quote() 
    except Exception as e:
        print(f"Error Loading Data: {e}")
        return

    # 2. 因子计算
    fe = FactorEngine(df_univ, cache_dir=f'./temp_factors_{universe_name}')
    df_fac = fe.run()
    if df_fac.empty: return
    del fe; gc.collect()

    # 3. 合并数据基础
    print(">>> [Data] Merging Data and Control Variables...")
    df_full = pd.merge(df_fac, df_univ.rename(columns={'date':'TradingDay', 'symbol':'SecuCode'}), on=['TradingDay', 'SecuCode'], how='inner')
    
    # 【修复核心】：补全 ClosePrice 用于计算反转和波动，修改 Volume 和 Amount 的列名
    target_cols = ['ClosePrice', 'Volume', 'Amount', 'NonRestrictedShares', 'AFloats']
    missing_cols = [c for c in target_cols if c not in df_full.columns and c in df_daily.columns]
    
    if missing_cols:
        df_supp = df_daily[['TradingDay', 'SecuCode'] + missing_cols]
        df_full = pd.merge(df_full, df_supp, on=['TradingDay', 'SecuCode'], how='left')

    del df_fac, df_univ; gc.collect()
    
    # 合并行业和市值
    df_ind, df_cap = DataLoader.load_industry_cap()
    df_full = df_full.sort_values('TradingDay')
    df_full = pd.merge_asof(df_full, df_ind[['SecuCode', 'InfoPublDate', 'tag']], left_on='TradingDay', right_on='InfoPublDate', by='SecuCode', direction='backward')
    df_full['Industry'] = df_full['tag'].fillna('Unknown')
    
    cols_cap = ['TradingDay', 'SecuCode', 'LnCap']
    df_full = pd.merge(df_full, df_cap[cols_cap], on=['TradingDay', 'SecuCode'], how='inner')
    
    # 4. 特征工程 (仅做去极值、标准化、正交化)
    df_final = process_features(df_full)

    # 5. 全面的因子分层与相关性分析
    plot_factor_correlation(df_final, method='spearman', title='Factor Correlation Matrix (Orthogonalized)')

    if not df_final.empty:
        # 直接传入清洗好的因子 df_final 和 行情数据 df_daily 
        run_factor_analysis_step(df_final, df_daily)

if __name__ == "__main__":
    if not os.path.exists(Config.ROOT_DIR):
        print(f"Error: Data directory '{Config.ROOT_DIR}' not found.")
    else:
        # run_pipeline(Config.PATH_INDEX_WT_300, "CSI300")
        run_pipeline(Config.PATH_INDEX_WT_500, "CSI500")