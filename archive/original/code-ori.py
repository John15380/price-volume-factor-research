import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from numba import jit
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from datetime import timedelta
from joblib import Parallel, delayed
import gc
import shutil # 用于清理临时文件夹
from tqdm import tqdm

# 设置绘图风格
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. 配置 (Config)
# -----------------------------------------------------------------------------
class Config:
    # --- 路径配置 (请确保这些路径与你的环境一致) ---
    ROOT_DIR = './data'
    
    # 基础数据路径
    PATH_HF_BAR = os.path.join(ROOT_DIR, 'bar1m_2016_2018') 
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
    ROLLING_WINDOW = 12   # 训练窗口(月)
    START_PREDICT = '2017-01-01'
    
    # --- 交易参数 ---
    TOP_K = 50            # 持仓数量
    COST_RATE_TOTAL = 0.003  # 双边千三
    MAX_GAP_DAYS = 20     # 【新增】超过20天无数据视为断层，清空历史

    FREQ_DAYS = {
        'Weekly': 5,
        'BiWeekly': 10,
        'Monthly': 20
    }

# -----------------------------------------------------------------------------
# 2. 核心计算函数 (Numba Core) - 保持方案A
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
    
    if var_x > 1e-12 and var_y > 1e-12:
        return cov / np.sqrt(var_x * var_y)
    return 0.0

@jit(nopython=True, cache=True)
def calc_daily_metrics_intraday(open_arr, close_arr, high_arr, low_arr, vol_arr):
    """
    Level 1: 计算基于分钟数据的当日基础指标
    修正：SmartMoney 过滤 0 成交量；CDPDP 差分相关性。
    注意：Index 6 (Day_Ret) 此处仅计算日内收益作为占位，
          真实的 Total Return 将在 FactorEngine 中结合昨日收盘价进行修正。
    """
    n = len(close_arr)
    if n < 30: return np.array([np.nan] * 7)
    
    # --- 1. Smart Money (保持过滤逻辑) ---
    s_metric = np.zeros(n)
    valid_sm_count = 0
    
    # 从 i=1 开始计算日内分钟收益，避免隔夜跳空干扰高频指标
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
                
                # 近似 VWAP
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

    # --- 2. CPV Component (Raw) ---
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
        
    # --- 3. CDPDP (Raw Sum) ---
    diffs = close_arr[1:] - close_arr[:-1]
    if len(diffs) > 10:
        x_seq = diffs[:-1]
        y_seq = diffs[1:]
        corr_up_up = calc_subset_correlation(x_seq, y_seq, 1)
        corr_down_down = calc_subset_correlation(x_seq, y_seq, 2)
        cdpdp_raw = corr_up_up + corr_down_down
    else:
        cdpdp_raw = np.nan
    
    # --- 4. Daily Basic Stats ---
    day_high = np.max(high_arr)
    day_low = np.min(low_arr)
    day_close = close_arr[-1]
    
    # 默认使用 Open 计算日内收益 (作为第一天的 fallback)
    # 真正的 Total Return 会在外部被覆盖
    if open_arr[0] > 1e-6:
        day_ret_intraday = day_close / open_arr[0] - 1.0 
    else:
        day_ret_intraday = 0.0
    
    return np.array([sm_factor, corr_pv, cdpdp_raw, day_high, day_low, day_close, day_ret_intraday])

# -----------------------------------------------------------------------------
# 3. 数据加载器 (Data Loader)
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

    @staticmethod
    def construct_label(df_daily, is_for_backtest=False):
        """
        修正版：
        1. Reindex 修复停牌时间穿越。
        2. 严格剔除 T+1 一字涨停和停牌样本，防止模型作弊。
        """
        print(f">>> [Loader] 构造 Labels (Strict Limit Check: {'Backtest' if is_for_backtest else 'Train'})...")
        
        # --- 1. 加载全量交易日历 (防止时间断层) ---
        df_cal = pd.read_csv(Config.PATH_TRADING_DAYS)
        df_cal['TradingDay'] = pd.to_datetime(df_cal['TradingDay'])
        
        min_date = df_daily['TradingDay'].min()
        max_date = df_daily['TradingDay'].max()
        all_trading_days = df_cal[(df_cal['TradingDay'] >= min_date) & 
                                  (df_cal['TradingDay'] <= max_date)]['TradingDay'].sort_values().values
        
        # --- 2. Reindex 全市场索引 ---
        all_codes = df_daily['SecuCode'].unique()
        idx = pd.MultiIndex.from_product([all_codes, all_trading_days], names=['SecuCode', 'TradingDay'])
        
        # 停牌日或缺失日会自动填充 NaN
        df_full = df_daily.set_index(['SecuCode', 'TradingDay']).reindex(idx)
        
        # --- 3. 计算 Shift (Label) ---
        g = df_full.groupby('SecuCode')
        
        # T+1 的状态
        df_full['Next_Open'] = g['OpenPrice'].shift(-1)
        df_full['Next_PTFlag'] = g['PTFlag'].shift(-1) # T+1 的停牌状态
        
        # T+2 的状态 (用于计算持有收益)
        df_full['Next_Next_Open'] = g['OpenPrice'].shift(-2)
        
        # 真正的结算日期 (T+2 日期)
        df_full['Temp_Date'] = df_full.index.get_level_values('TradingDay')
        df_full['SettlementDate'] = df_full.groupby('SecuCode')['Temp_Date'].shift(-2)
        df_full.drop(columns=['Temp_Date'], inplace=True)

        # 重置索引，方便向量化计算
        df = df_full.reset_index()
        
        # --- 4. 【核心新增】严格的涨跌停阈值计算 ---
        # 目的：判断 T+1 开盘是否一字涨停 (无法买入)
        # 逻辑：
        #   - 科创板 (688): 20%
        #   - 北交所 (8xx/4xx): 30%
        #   - 创业板 (300): 2020-08-24 前 10%, 后 20%
        #   - 其他 (主板): 10% (ST暂统一按10%处理或需额外ST列表，此处偏向保守取 9.8% 左右作为阈值)
        
        # 提取 Code 前缀和日期
        codes_series = df['SecuCode'].astype(str)
        dates = df['TradingDay'].values
        
        # 构造条件掩码 (使用 Pandas 原生向量化操作，会自动处理底层类型)
        # .values 将 Series 转为 numpy boolean array，性能无损
        is_688 = codes_series.str.startswith('688').values
        is_bj = (codes_series.str.startswith('8') | codes_series.str.startswith('4')).values
        is_300 = codes_series.str.startswith('300').values
        
        date_threshold = np.datetime64('2020-08-24')
        is_after_reform = dates >= date_threshold
        
        # 向量化选择阈值 (使用稍微收紧的阈值以防精度误差)
        limit_thresholds = np.select(
            [
                is_bj,                            # 北交所 30%
                is_688,                           # 科创板 20%
                (is_300 & is_after_reform),       # 创业板注册制后 20%
                (is_300 & ~is_after_reform)       # 创业板注册制前 10%
            ],
            [0.29, 0.195, 0.195, 0.095],          # 对应的值
            default=0.095                         # 主板默认 10%
        )
        
        # 存储阈值供后续判断
        df['Limit_Threshold'] = limit_thresholds
        
        # --- 5. 计算收益与 Label 清洗 ---
        # T+1 开盘涨幅 (相对于 T 日收盘价)
        # 注意：这里是用 T+1 Open / T Close - 1 来判断是否在 T+1 开盘时就涨停
        df['Next_Open_Chg'] = df['Next_Open'] / df['ClosePrice'] - 1.0
        
        # 计算持有收益 (T+1 Open 到 T+2 Open)
        df['Ret_Hold'] = df['Next_Next_Open'] / df['Next_Open'] - 1.0
        
        if not is_for_backtest:
            # === [训练模式 Training Mode] ===
            # 必须剔除模型无法交易的样本，防止 Label Leakage
            
            # 条件A: T+1 开盘一字涨停 (买不进去)
            cond_limit_up = df['Next_Open_Chg'] >= df['Limit_Threshold']
            
            # 条件B: T+1 停牌 (PTFlag != 0, 买不进去)
            # 注意：Reindex 后，如果当天没数据，Next_PTFlag 也是 NaN，这也会被下面的 notna() 过滤掉
            # 这里假设 PTFlag=0 是正常，其他是异常
            cond_suspend = (df['Next_PTFlag'] != 0) | df['Next_PTFlag'].isna()
            
            # 将这些“不可交易”样本的收益设为 NaN，从而在 dropna 时剔除
            mask_invalid = cond_limit_up | cond_suspend
            df.loc[mask_invalid, 'Ret_Hold'] = np.nan
            
            # 截断极端值
            df['Ret_Hold'] = df['Ret_Hold'].clip(-0.2, 0.2)
            
            # 丢弃缺失值 (包含原始缺失、停牌、一字涨停)
            df = df.dropna(subset=['Ret_Hold', 'ClosePrice'])
            
        else:
            # === [回测模式 Backtest Mode] ===
            # 保留所有行（包括 NaN），以便回测引擎查询“当天是否停牌”
            pass
            
        return df

# -----------------------------------------------------------------------------
# 4. 因子引擎 (Factor Engine) - 【修正：Time Gap Logic】
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
        
        # 160天回溯 + 10天Buffer
        self.MAX_HISTORY = 170
        # Index: 0:SM, 1:CorrPV, 2:CDPDP, 3:High, 4:Low, 5:Close, 6:Ret
        self.history = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
        
        # 防止长假清空 Buffer
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
        print(f">>> [Factor] 开始计算高频与滚动因子 (Final Fixed Version)...")
        print(f"    History Window: {self.MAX_HISTORY} days")
        
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
                
                # 1. 计算当日分钟级 Metrics
                daily_mets = calc_daily_metrics_intraday(
                    sub_vals['open'].values, sub_vals['close'].values,
                    sub_vals['high'].values, sub_vals['low'].values,
                    sub_vals['volume'].values
                )
                
                # 2. 状态更新 (Deque) & 修正收益率口径
                queue = self.history[sym]
                
                if len(queue) > 0:
                    last_date, last_mets = queue[-1]
                    
                    # 检查时间断层
                    if (curr_date - last_date).days > self.MAX_GAP_DAYS:
                        queue.clear()
                    else:
                        # ===【核心修正 1】计算真实收益率 (Total Return) ===
                        prev_close = last_mets[5] # Index 5 is Close
                        curr_close = daily_mets[5]
                        
                        if prev_close > 1e-6:
                            # 覆盖 Index 6 (原为 Close/Open - 1)
                            true_ret = curr_close / prev_close - 1.0
                            daily_mets[6] = true_ret
                        else:
                            daily_mets[6] = 0.0

                # 存入当日数据 (只要 Close 有效)
                if not np.isnan(daily_mets[5]): 
                    queue.append((curr_date, daily_mets))
                
                # 3. 因子构造 (Rolling Factors)
                n_hist = len(queue)
                if n_hist < 20: continue 
                
                hist_data = np.array([item[1] for item in queue]) 
                
                # --- Factor 1: Smart Money ---
                sm_vals = hist_data[-10:, 0]
                sm_vals = sm_vals[sm_vals > -0.5] 
                val_sm = np.mean(sm_vals) if len(sm_vals) > 0 else np.nan
                
                # --- Factor 2: CPV (Raw) ---
                cpv_window = 20
                val_cpv_mean, val_cpv_std, val_cpv_trend = np.nan, np.nan, np.nan
                if n_hist >= cpv_window: 
                    cpv_slice = hist_data[-cpv_window:, 1]
                    mask = ~np.isnan(cpv_slice)
                    if np.sum(mask) > 15:
                        clean_cpv = cpv_slice[mask]
                        val_cpv_mean = -1.0 * np.mean(clean_cpv)
                        val_cpv_std = -1.0 * np.std(clean_cpv)
                        val_cpv_trend = -1.0 * calc_linear_slope(clean_cpv)

                # --- Factor 3: CDPDP' ---
                cdpdp_slice = hist_data[-20:, 2] 
                cdpdp_slice = cdpdp_slice[~np.isnan(cdpdp_slice)]
                val_cdpdp = -1 * np.mean(cdpdp_slice) if len(cdpdp_slice) > 0 else np.nan
                
                # --- Factor 4: Ideal Amplitude (理想振幅) ---
                val_ideal_amp = np.nan
                if n_hist >= 20:
                    sub_h = hist_data[-20:, 3]
                    sub_l = hist_data[-20:, 4]
                    sub_c = hist_data[-20:, 5]
                    
                    # ===【核心修正 2】数据清洗: 确保 H >= L ===
                    # 应对数据源脏数据或极端情况
                    sub_h = np.maximum(sub_h, sub_l)
                    
                    valid_amp = sub_l > 1e-6
                    if np.sum(valid_amp) > 15:
                        amps = (sub_h[valid_amp] / sub_l[valid_amp]) - 1.0
                        closes = sub_c[valid_amp]
                        
                        sort_indices = np.argsort(closes) 
                        k = max(1, int(len(closes) * 0.25))
                        
                        v_low = np.mean(amps[sort_indices[:k]])
                        v_high = np.mean(amps[sort_indices[-k:]])
                        val_ideal_amp = -1.0 * (v_high - v_low)

                # --- Factor 5: Long-term Momentum (修正窗口 + 收益口径) ---
                val_mom_long = np.nan
                target_window = 160
                
                if n_hist >= target_window:
                    sub_h = hist_data[-target_window:, 3] 
                    sub_l = hist_data[-target_window:, 4]
                    
                    # 注意：此处使用的 hist_data[:, 6] 已经是修正后的 Total Return
                    sub_ret = hist_data[-target_window:, 6]
                    
                    # 同样做 H >= L 保护
                    sub_h = np.maximum(sub_h, sub_l)
                    
                    valid_mom = sub_l > 1e-6
                    if np.sum(valid_mom) > 140:
                        amps = (sub_h[valid_mom] / sub_l[valid_mom]) - 1.0
                        rets = sub_ret[valid_mom]
                        
                        amp_sort_idx = np.argsort(amps) 
                        cutoff = int(len(amps) * 0.70)
                        
                        # 取低波动日子的收益率之和
                        target_idx = amp_sort_idx[:cutoff]
                        val_mom_long = np.sum(rets[target_idx])
                
                # 4. 写入 Buffer
                self.buffer.append({
                    'TradingDay': curr_date,
                    'SecuCode': str(sym).zfill(6),
                    'HF_SmartMoney': val_sm,
                    'HF_Raw_CPV_Mean': val_cpv_mean, 
                    'HF_Raw_CPV_Std': val_cpv_std,
                    'HF_Raw_CPV_Trend': val_cpv_trend,
                    'HF_CDPDP': val_cdpdp,
                    'HF_IdealAmp': val_ideal_amp,
                    'HF_LongTermMom': val_mom_long
                })
            
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
                
            # 用 set_postfix 显示当前日期
            file_iterator.set_postfix(date=date_str)

        self._flush_buffer()
        print(">>> [Factor] 计算完成，正在合并所有分片...")
        try:
            df_final = pd.read_parquet(self.cache_dir)
            return df_final
        except Exception as e:
            print(f"Error merging parquet files: {e}")
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# 5. 数据处理与特征工程 (Updated: CPV Neutralization)
# -----------------------------------------------------------------------------
def process_features_and_label(df_full, temp_dir='./temp_processed_features'):
    """
    内存优化版数据预处理：
    1. 预计算控制变量 (Reversal, Volatility, Turnover) 用于 CPV 正交化。
    2. Label 行业中性化。
    3. 特征清洗与正交化 (分批处理 + 差异化回归)。
    """
    print(f">>> [Process] 执行特征工程 (Strict Neutralization Mode)...")
    
    # --- 0. 环境清理 ---
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 确保数据按时间排序，方便计算 Rolling
    df_full = df_full.sort_values(['SecuCode', 'TradingDay'])

    # --- 1. 预计算控制变量 (Control Factors) ---
    # 这些变量用于剔除 CPV 因子中的风格干扰
    print("    [1/4] 计算风格控制变量 (Reversal, Volatility, Turnover)...")
    
    # A. Reversal (Ret20): 过去20日收益
    # 使用 ClosePrice 计算
    df_full['Ctrl_Ret20'] = df_full.groupby('SecuCode')['ClosePrice'].pct_change(20)
    
    # B. Volatility (Vol20): 过去20日日收益率的标准差
    df_full['Ctrl_Vol20'] = df_full.groupby('SecuCode')['ClosePrice'].transform(
        lambda x: x.pct_change().rolling(20).std()
    )
    
    # C. Turnover (Turnover20): 过去20日平均换手率 (需要 Volume 和 Cap)
    # 如果数据中没有 Volume，则跳过此项
    has_turnover_data = False
    # 1. 映射列名（确保后续代码能找到）
    # 即使你的 CSV 叫 TradingVolumes，为了通用性，我们建议新建标准列名
    if 'TradingVolumes' in df_full.columns:
        df_full['Volume'] = df_full['TradingVolumes'] # 对应成交量(股)
    if 'TurnoverValue' in df_full.columns:
        df_full['Amount'] = df_full['TurnoverValue']  # 对应成交额

    # 2. 确定流通股本 (优先用非限售，缺失则用A股流通)
    # 注意：分母不能为 0
    if 'FloatShares' not in df_full.columns:
        if 'NonRestrictedShares' in df_full.columns:
            print("    [Info] Constructing 'FloatShares' from 'NonRestrictedShares'...")
            df_full['FloatShares'] = df_full['NonRestrictedShares']
            
            # 如果有 AFloats，用它填充非限售股本的空缺
            if 'AFloats' in df_full.columns:
                df_full['FloatShares'] = df_full['FloatShares'].fillna(df_full['AFloats'])
                
        elif 'AFloats' in df_full.columns:
            print("    [Info] Constructing 'FloatShares' from 'AFloats'...")
            df_full['FloatShares'] = df_full['AFloats']
        else:
            print("    [Error] Neither 'NonRestrictedShares' nor 'AFloats' found!")

    # 3. 计算换手率逻辑
    # 只要有 Volume 和 FloatShares 即可

    print("Debug Columns:", df_full.columns.tolist())
    
    if 'Volume' in df_full.columns and 'FloatShares' in df_full.columns:
        # 确保分母不为 0
        valid_cap = df_full['FloatShares'] > 1e-6
        
        # 初始化
        df_full['Turnover_Daily'] = np.nan
        
        # 计算：换手率 = 成交量 / 流通股本
        # 注意：你的 TradingVolumes 是 24,089,142 (股)，Shares 是 3,105,358,511 (股)
        # 直接相除即可。不需要乘以 100，除非你的 Volume 单位是"手"
        # 看你的示例数据，TradingVolumes 看起来是"股" (2400万股对应3.7亿金额，均价15块，合理)
        df_full.loc[valid_cap, 'Turnover_Daily'] = (
            df_full.loc[valid_cap, 'Volume'] / df_full.loc[valid_cap, 'FloatShares']
        )
        
        # 极值处理：换手率超过 200% 或小于 0 的通常是数据错误，可以清洗一下（可选）
        # df_full.loc[df_full['Turnover_Daily'] > 2.0, 'Turnover_Daily'] = 2.0 
        
        # 4. 计算 20日平滑换手率 (用于中性化)
        # 必须按股票分组计算 Rolling
        df_full['Ctrl_Turnover20'] = df_full.groupby('SecuCode')['Turnover_Daily'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        
        # 填充一下初始的 NaN (用当天的替代，避免整行数据被丢弃)
        df_full['Ctrl_Turnover20'].fillna(df_full['Turnover_Daily'], inplace=True)
        
        print("    [Info] Turnover calculated using NonRestrictedShares/AFloats.")
        
        # 标记 flag 供后续中性化模块使用
        # 注意：后续代码可能在检查 'Cap' 或 'Volume'，确保你后续的中性化代码能读到 'Ctrl_Turnover20'
        has_turnover_data = True 

    else:
        print("    Warning: Still missing Volume or Shares columns. Turnover Check Failed.")
        has_turnover_data = False

    # 填充控制变量的 NaN (无法计算的前20天填 0，避免整行被删)
    ctrl_cols = ['Ctrl_Ret20', 'Ctrl_Vol20']
    if has_turnover_data: ctrl_cols.append('Ctrl_Turnover20')
    df_full[ctrl_cols] = df_full[ctrl_cols].fillna(0)

    # --- 2. Label 行业中性化 ---
    if 'Industry' not in df_full.columns:
        df_full['Industry'] = 'Unknown'
    
    print("    [2/4] 计算 Label 行业相对收益...")
    ind_means = df_full.groupby(['TradingDay', 'Industry'])['Ret_Hold'].transform('mean')
    df_full['Ret_Excess'] = df_full['Ret_Hold'] - ind_means
    df_full['Ret_Excess'] = df_full['Ret_Excess'].fillna(0)
    df_full['Ret_Excess'] = df_full.groupby('TradingDay')['Ret_Excess'].transform(lambda x: x - x.mean())

    # --- 3. 特征清洗与差异化中性化 (分批处理) ---
    feature_cols = [c for c in df_full.columns if c.startswith('HF_')]
    cpv_cols = [c for c in feature_cols if 'CPV' in c] # 识别 CPV 因子
    normal_cols = [c for c in feature_cols if 'CPV' not in c]

    def _process_day_logic(sub_df):
        # 1. 剔除无效样本
        sub_df = sub_df.dropna(subset=feature_cols, how='all')
        if len(sub_df) < 20: return pd.DataFrame() 
        
        # 2. 缺失值填充 (中位数)
        for col in feature_cols:
            median_val = sub_df[col].median()
            sub_df[col] = sub_df[col].fillna(0 if pd.isna(median_val) else median_val)
        
        # 3. 构造回归矩阵
        # 基础 X: 行业 + 市值
        X_ind = pd.get_dummies(sub_df['Industry'], prefix='Ind')
        if X_ind.shape[1] > 0:
            X_base = pd.concat([X_ind, sub_df[['LnCap']]], axis=1).astype(float)
        else:
            X_base = sub_df[['LnCap']].astype(float)
            
        # 扩展 X (用于 CPV): 行业 + 市值 + 反转 + 波动 + 换手
        # 注意要对齐索引
        X_extended = pd.concat([X_base, sub_df[ctrl_cols]], axis=1).astype(float)
        
        X_base = X_base.fillna(0)
        X_extended = X_extended.fillna(0)
            
        model = LinearRegression()
        
        # 4. 逐个因子处理
        for col in feature_cols:
            y = sub_df[col].values
            
            # MAD 去极值
            median = np.median(y)
            diff = np.abs(y - median)
            mad = np.median(diff)
            mad = np.mean(diff) if mad == 0 else mad
            y = np.clip(y, median - 5 * mad, median + 5 * mad)
            
            # --- 差异化中性化 ---
            if col in cpv_cols:
                # CPV 因子：严格剔除所有风格
                if len(X_extended) > 0:
                    model.fit(X_extended, y)
                    y = y - model.predict(X_extended)
            else:
                # 普通因子：仅剔除行业和市值
                if len(X_base) > 0:
                    model.fit(X_base, y)
                    y = y - model.predict(X_base)
            
            # Z-Score 标准化
            std_val = np.std(y)
            sub_df[col] = 0 if std_val == 0 else (y - np.mean(y)) / (std_val + 1e-9)
                
        return sub_df

    print(f"    [3/4] 分批特征处理 (Batch -> Orthogonalization -> Disk)...")
    
    # 按月分块处理
    df_full['Month'] = df_full['TradingDay'].dt.to_period('M')
    unique_months = df_full['Month'].unique()
    
    for i, month in enumerate(tqdm(unique_months, desc="Processing Batches")):
        batch_df = df_full[df_full['Month'] == month].copy()
        
        daily_groups = [group for _, group in batch_df.groupby('TradingDay')]
        
        processed_list = Parallel(n_jobs=Config.N_JOBS)(
            delayed(_process_day_logic)(day_df) for day_df in daily_groups
        )
        
        processed_list = [res for res in processed_list if not res.empty]
        if processed_list:
            batch_result = pd.concat(processed_list)
            
            # ---【修复核心 1】筛选最终列 ---
            # 确保只保留需要的列，去除中间变量
            keep_cols = ['TradingDay', 'SecuCode', 'LnCap', 'Industry', 
                         'Ret_Hold', 'Ret_Excess', 'Next_PTFlag', 'Next_Open_Chg']
            
            # 动态添加因子列 (feature_cols) 和 控制变量列 (如果存在)
            potential_cols = feature_cols + ctrl_cols + ['Volume', 'Amount', 'Turnover_Daily']
            for c in potential_cols:
                if c in batch_result.columns:
                    keep_cols.append(c)
            
            # 确保列存在
            final_cols = [c for c in keep_cols if c in batch_result.columns]
            batch_result = batch_result[final_cols]
            
            # ---【修复核心 2】强制类型转换，防止 Int/Float 推断冲突 ---
            # 排除掉 非数值列
            exclude_cols = ['TradingDay', 'SecuCode', 'Industry']
            
            # 找出所有需要转 float 的列
            cols_to_convert = [c for c in batch_result.columns if c not in exclude_cols]
            
            # 强制转为 float32 (既省内存，又避免 0 被存为 Int)
            # 使用 errors='ignore' 防止万一有奇怪的非数值列
            for col in cols_to_convert:
                try:
                    batch_result[col] = batch_result[col].astype('float32')
                except:
                    pass # 如果转换失败(比如是字符串)，就跳过

            # 保存
            save_path = os.path.join(temp_dir, f"batch_{month}.parquet")
            batch_result.to_parquet(save_path, index=False)
        
        del batch_df, daily_groups, processed_list
        gc.collect()

    print("    [4/4] 合并最终数据集...")
    try:
        df_final = pd.read_parquet(temp_dir)
        df_final = df_final.sort_values(['TradingDay', 'SecuCode'])
        if 'Industry' in df_final.columns:
            df_final['Industry'] = df_final['Industry'].astype('category')
        return df_final
    except Exception as e:
        print(f"Error merging processed files: {e}")
        return pd.DataFrame()

def train_model(df_data):
    # 保持原代码逻辑
    print(">>> [Model] 开始滚动训练...")
    feature_cols = [c for c in df_data.columns if c.startswith('HF_')] + ['LnCap', 'Industry']
    target_col = 'Ret_Excess'
    all_dates = sorted(df_data['TradingDay'].unique())
    months = sorted(list(set([pd.Timestamp(d).to_period('M') for d in all_dates])))
    preds_list = []
    
    start_period = pd.Period(Config.START_PREDICT, 'M')
    try: start_idx = months.index(start_period)
    except: start_idx = Config.ROLLING_WINDOW

    iterator = tqdm(range(start_idx, len(months)), desc="[Model] Rolling Walk-Forward")
        
    for i in iterator:
        test_month = months[i]
        test_mask = df_data['TradingDay'].dt.to_period('M') == test_month
        test_dates = df_data.loc[test_mask, 'TradingDay'].unique()
        if len(test_dates) == 0: continue
        
        test_start_date = test_dates.min()
        available_dates = [d for d in all_dates if d < test_start_date]
        train_end_idx = len(available_dates) - 3
        if train_end_idx < Config.ROLLING_WINDOW * 20: continue 
        train_end_date = available_dates[train_end_idx]
        train_start_month = months[i - Config.ROLLING_WINDOW]
        
        train_mask = (df_data['TradingDay'].dt.to_period('M') >= train_start_month) & \
                     (df_data['TradingDay'] <= train_end_date) & \
                     (df_data[target_col].notna())
        
        X_train = df_data.loc[train_mask, feature_cols]
        y_train = df_data.loc[train_mask, target_col]
        X_test = df_data.loc[test_mask, feature_cols]
        y_test = df_data.loc[test_mask, target_col]
        meta_cols = ['TradingDay', 'SecuCode', 'Ret_Hold', 'Next_PTFlag', 'Next_Open_Chg']
        meta_test = df_data.loc[test_mask, meta_cols]
        
        if len(X_train) < 100 or len(X_test) == 0: continue
        
        model = xgb.XGBRegressor(
            max_depth=5, learning_rate=0.05, n_estimators=100,
            n_jobs=4, random_state=42, objective='reg:squarederror',
            enable_categorical=True, tree_method='hist'
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        df_tmp_metric = pd.DataFrame({
            'TradingDay': meta_test['TradingDay'].values,
            'True': y_test.values,
            'Pred': pred
        })
        
        # ===【新增优化 4】计算 IC (Rank IC) ===
        # 衡量当期预测值与真实值的相关性
        # 如果 IC 长期 < 0.02，说明模型没效
        curr_ic = np.corrcoef(pred, y_test)[0, 1]
        daily_ic = df_tmp_metric.groupby('TradingDay').apply(
            lambda x: x['Pred'].corr(x['True'], method='spearman')
        )
        rank_ic = daily_ic.mean()
        
        msg = f"Predict: {test_month} | IC: {curr_ic:.3f} | RankIC: {rank_ic:.3f}"
        tqdm.write(msg)

        res = meta_test.copy()
        res['Pred_Score'] = pred
        res['Pred_Rank'] = res.groupby('TradingDay')['Pred_Score'].rank(ascending=False)
        preds_list.append(res)
        
    if not preds_list: return pd.DataFrame()
    return pd.concat(preds_list)

# -----------------------------------------------------------------------------
# 6. 回测引擎 (Hedged Backtest)
# -----------------------------------------------------------------------------
class HedgedBacktest:
    def __init__(self, df_preds, df_daily_quote, universe_name):
        self.df = df_preds.sort_values(['TradingDay', 'SecuCode'])
        self.daily_quote = df_daily_quote.set_index(['TradingDay', 'SecuCode']).sort_index()
        self.name = universe_name
        self.cost_rate = Config.COST_RATE_TOTAL
        self.top_k = Config.TOP_K
        self.buffer_size = max(1, int(self.top_k * 0.4))
        self.rebalance_threshold = 0.1 
        # Top 50 持仓，但只有跌出 Top 80 才卖出，只有进入 Top 30 才买入
        self.buy_threshold = 30  
        self.sell_threshold = 80

    def _get_dynamic_threshold(self, code, date_ts):
        # 辅助函数保持不变
        if code.startswith('688'): return 0.195
        if code.startswith('8') or code.startswith('4'): return 0.295
        if code.startswith('300'):
            return 0.195 if date_ts >= pd.Timestamp('2020-08-24') else 0.095
        return 0.095

    def run(self):
        print(f">>> [Backtest] 开始多空回测 (Fixed: Strict Suspension Logic)...")
        
        signal_dates = sorted(self.df['TradingDay'].unique())
        stats = []
        
        current_equity = 1.0
        current_holdings = {} # {code: weight}
        
        iterator = tqdm(signal_dates, desc="[Backtest] Daily Sim")

        for date in iterator:
            # --- 1. 准备当日数据 ---
            daily_preds = self.df[self.df['TradingDay'] == date]
            
            # 安全获取 SettlementDate
            try:
                if date in self.daily_quote.index.get_level_values(0):
                    day_slice = self.daily_quote.loc[date]
                    valid_dates = day_slice['SettlementDate'].dropna()
                    if not valid_dates.empty:
                        settlement_date = valid_dates.iloc[0]
                    else:
                        continue 
                else:
                    continue
            except:
                continue
                
            daily_preds = daily_preds.sort_values('Pred_Rank', ascending=True)
            rank_map = daily_preds.set_index('SecuCode')['Pred_Rank'].to_dict()
            
            # --- 2. 生成目标持仓 (Buffer逻辑) ---
            keep_list = []
            current_codes = list(current_holdings.keys())
            
            for code in current_codes:
                rank = rank_map.get(code, 99999) # 没预测到的股票排最后
                if rank <= self.sell_threshold:
                    keep_list.append(code)
            
            n_needed = self.top_k - len(keep_list)
            if n_needed > 0:
                candidates = daily_preds[
                    (~daily_preds['SecuCode'].isin(keep_list)) & 
                    (daily_preds['Pred_Rank'] <= self.buy_threshold) # 严格买入
                ]
                new_buys = candidates.head(n_needed)['SecuCode'].values
                target_codes = set(keep_list) | set(new_buys)
            else:
                keep_list_sorted = sorted(keep_list, key=lambda c: rank_map.get(c, 99999))
                target_codes = set(keep_list_sorted[:self.top_k])
            
            # --- 3. 交易意图计算 ---
            final_count = len(target_codes)
            ideal_w_single = 1.0 / final_count if final_count > 0 else 0.0
            
            # 分离两类持仓：可交易(Liquid) 和 不可交易(Locked)
            locked_holdings = {} # {code: weight}
            liquid_holdings = {} # {code: weight}
            
            all_involved = set(current_holdings.keys()) | target_codes
            
            # 临时存储下一步的各种状态
            temp_next_state = {} 

            for code in all_involved:
                old_w = current_holdings.get(code, 0.0)
                is_target = code in target_codes
                ideal_w = ideal_w_single if is_target else 0.0
                
                # 获取交易状态
                can_buy = False
                can_sell = False
                next_ret = 0.0
                limit_threshold = 0.1 # Default
                
                if (date, code) in self.daily_quote.index:
                    info = self.daily_quote.loc[(date, code)]
                    pt_flag = info.get('Next_PTFlag', 1)
                    open_chg = info.get('Next_Open_Chg', 0.0)
                    next_ret = info.get('Ret_Hold', 0.0)
                    if pd.isna(next_ret): next_ret = 0.0
                    
                    if 'Limit_Threshold' in info:
                        limit_threshold = info['Limit_Threshold']
                    else:
                        limit_threshold = self._get_dynamic_threshold(code, date)
                    
                    # 严格交易判定
                    if pd.notna(pt_flag) and pt_flag == 0:
                        if pd.notna(open_chg) and open_chg < limit_threshold - 0.001: # 留点buffer
                            can_buy = True
                        if pd.notna(open_chg) and open_chg > -limit_threshold + 0.001:
                            can_sell = True
                
                # 记录该股票的属性，供后续归一化使用
                temp_next_state[code] = {
                    'old_w': old_w,
                    'ideal_w': ideal_w,
                    'can_buy': can_buy,
                    'can_sell': can_sell,
                    'ret': next_ret,
                    'is_target': is_target
                }

                # 判定是否被锁仓 (Locked)
                # 锁仓条件：想卖(权重变小)但不能卖
                if ideal_w < old_w and not can_sell:
                    locked_holdings[code] = old_w # 被迫锁仓，维持原权重
                # 注意：想买但不能买的情况，不算锁仓，因为资金还在手上，可以分配给别人
                
            # --- 4. 资金归一化 (核心修复) ---
            # 逻辑：总资金 1.0 = Locked_Weight + Liquid_Available
            total_locked_w = sum(locked_holdings.values())
            remaining_cap = 1.0 - total_locked_w
            
            if remaining_cap < 0:
                # 极端情况：锁仓股涨太多，超过总资产（透支），实际上此时你什么都做不了
                # 只能接受这个权重
                remaining_cap = 0.0 
            
            # 计算 Liquid 部分的目标总权重
            # 我们需要把 remaining_cap 分配给那些 'can_buy' 或者 'can_sell' 成功的股票
            
            actual_holdings_next = locked_holdings.copy() # 先把锁仓的放进去
            
            # 筛选出需要分配资金的 Liquid 股票
            liquid_targets = []
            for code, state in temp_next_state.items():
                if code in locked_holdings: continue
                
                # 如果是原有持仓，且能卖，或者 目标持仓且能买
                # 这里简化处理：按 ideal_w 分配，但要归一化到 remaining_cap
                if state['is_target']:
                    if state['can_buy'] or state['old_w'] > 0: # 能买进，或者虽然买不进但本来就有货(且没被锁)
                         liquid_targets.append(code)
            
            # 如果没有可买的票，剩下的钱就变成现金（不持有）
            if liquid_targets and remaining_cap > 0:
                # 简单的均分逻辑 (也可以按原比例缩放)
                # 这里我们尽量接近 ideal_w，但必须受制于 remaining_cap
                
                # 计算这些 liquid target 的原始 ideal_w 总和
                sum_ideal_liquid = sum(temp_next_state[c]['ideal_w'] for c in liquid_targets)
                
                if sum_ideal_liquid > 0:
                    scale = remaining_cap / sum_ideal_liquid
                else:
                    scale = 0 # Should not happen if logic is correct
                
                for code in liquid_targets:
                    state = temp_next_state[code]
                    
                    # 意图权重
                    w_intended = state['ideal_w'] * scale
                    
                    # 具体操作限制
                    final_w = 0.0
                    
                    if w_intended > state['old_w']: # 加仓
                        if state['can_buy']:
                            final_w = w_intended
                        else:
                            final_w = state['old_w'] # 买不进，维持原状
                    else: # 减仓
                        if state['can_sell']:
                            final_w = w_intended
                        else:
                            final_w = state['old_w'] # 实际上前面已经处理了 locked，这里应该不会进
                            
                    if final_w > 1e-6:
                        actual_holdings_next[code] = final_w

            # 再次检查：因为上面的 scale 是基于 ideal 计算的，但由于“买不进”的情况，
            # 实际分配的权重可能小于 remaining_cap (这就变成了现金保留)
            # 这一步不需要再强制归一化，因为我们不允许借钱（权重和 <= 1.0 是安全的）

            # --- 5. 结算与更新 ---
            turnover_value = 0.0
            gross_ret_weighted = 0.0
            holding_values_end = {}
            
            for code, final_w in actual_holdings_next.items():
                old_w = current_holdings.get(code, 0.0)
                turnover_value += abs(final_w - old_w)
                
                ret = temp_next_state[code]['ret']
                gross_ret_weighted += final_w * ret
                holding_values_end[code] = final_w * (1 + ret)
            
            # 如果实际持仓 < 1，剩下的现金收益为 0
            cash_w = max(0, 1.0 - sum(actual_holdings_next.values()))
            # gross_ret_weighted += cash_w * 0.0 
            
            cost = turnover_value * self.cost_rate
            net_ret = gross_ret_weighted - cost
            current_equity *= (1 + net_ret)
            
            # 漂移处理
            denominator = 1 + net_ret
            current_holdings = {}
            if denominator > 1e-9:
                for code, val in holding_values_end.items():
                    current_holdings[code] = val / denominator
            
            # 添加现金漂移 (如果不满仓)
            # 这里的 current_holdings sum 可能 < 1，这是对的

            stats.append({
                'SignalDate': date,
                'SettlementDate': settlement_date,
                'Equity': current_equity,
                'Turnover': turnover_value,
                'Count': len(actual_holdings_next),
                'DailyRet': net_ret
            })

            iterator.set_postfix(Eq=f"{current_equity:.2f}")
            
        return pd.DataFrame(stats)

# -----------------------------------------------------------------------------
# 7. 因子分析模块 (Factor Analysis Module)
# -----------------------------------------------------------------------------
class FactorAnalyzer:
    @staticmethod
    def calculate_metrics(daily_ret_series):
        """
        计算纯因子的绩效指标 (基于日频多空收益)
        """
        # 1. 年化收益 (单利累加近似，或复利)
        # 这里采用复利年化
        cum_ret = (1 + daily_ret_series).cumprod()
        total_ret = cum_ret.iloc[-1] - 1
        
        days = len(daily_ret_series)
        if days < 10: return {}
        
        # 年化收益率 (Geometric)
        ann_ret = (total_ret + 1) ** (250 / days) - 1
        
        # 2. 年化波动率
        ann_vol = daily_ret_series.std() * np.sqrt(250)
        
        # 3. 信息比率 (IR)
        if ann_vol == 0:
            ir = 0
        else:
            ir = ann_ret / ann_vol
            
        # 4. 最大回撤
        cum_max = cum_ret.cummax()
        drawdown = cum_ret / cum_max - 1
        max_dd = drawdown.min()
        
        # 5. 月度胜率
        # 将日收益重采样为月收益
        monthly_ret = daily_ret_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        win_months = len(monthly_ret[monthly_ret > 0])
        total_months = len(monthly_ret)
        win_rate = win_months / total_months if total_months > 0 else 0
        
        return {
            'Ann. Ret': ann_ret,
            'Ann. Vol': ann_vol,
            'IR': ir,
            'Max DD': max_dd,
            'Monthly Win': win_rate
        }

    @staticmethod
    def analyze_factor(df_data, factor_col, return_col='Ret_Hold', factor_name=None, ascending=True):
        """
        对单个因子进行分层测试
        :param df_data: 包含因子和收益率的 DataFrame
        :param factor_col: 因子列名
        :param return_col: 收益率列名 (通常用 Ret_Hold 代表 T+1 到 T+2 的收益)
        :param ascending: True=因子越大越好(Group5最高), False=因子越小越好
        """
        if factor_name is None: factor_name = factor_col
        print(f"--- Analyzing Factor: {factor_name} ---")
        
        # 1. 提取必要数据并去空
        data = df_data[['TradingDay', 'SecuCode', factor_col, return_col]].copy()
        data = data.dropna()
        
        if data.empty:
            print("  No valid data for this factor.")
            return

        # 2. 每日分层 (Quintiles)
        # 这里的 Group 0 是最小，Group 4 是最大
        def get_group(x):
            try:
                # qcut 可能因为数据重复报错，加上 duplicate='drop'
                return pd.qcut(x, 5, labels=False, duplicates='drop')
            except:
                return np.nan

        data['Group'] = data.groupby('TradingDay')[factor_col].transform(get_group)
        data = data.dropna(subset=['Group'])
        
        # 3. 计算各组每日平均收益
        group_ret = data.groupby(['TradingDay', 'Group'])[return_col].mean().unstack()
        # group_ret 形状: Index=Date, Cols=[0, 1, 2, 3, 4]
        
        # 如果 ascending=False (因子越小越好)，则 Group 0 是最好的，Group 4 是最差的
        # 为了统一“Group 5 是做多组”，如果 ascending=False，我们需要反转列
        if not ascending:
            group_ret = group_ret[group_ret.columns[::-1]]
            group_ret.columns = range(5) # 重置为 0-4
            
        # 4. 计算多空收益 (Top - Bottom)
        # 假设做多 Group 4 (Top 20%)，做空 Group 0 (Bottom 20%)
        long_short_ret = group_ret[4] - group_ret[0]
        
        # 5. 计算指标
        metrics = FactorAnalyzer.calculate_metrics(long_short_ret)
        
        # 6. 打印指标
        print(f"  Long-Short Metrics:")
        print(f"    Ann. Ret:    {metrics.get('Ann. Ret', 0):.2%}")
        print(f"    Ann. Vol:    {metrics.get('Ann. Vol', 0):.2%}")
        print(f"    IR:          {metrics.get('IR', 0):.2f}")
        print(f"    Max DD:      {metrics.get('Max DD', 0):.2%}")
        print(f"    Mo. WinRate: {metrics.get('Monthly Win', 0):.2%}")
        
        # 7. 绘图
        plt.figure(figsize=(10, 6))
        
        # 绘制 5 组净值曲线
        cum_groups = (1 + group_ret).cumprod()
        colors = plt.cm.RdYlGn(np.linspace(0, 1, 5)) # 红(差)->绿(好)
        
        for i in range(5):
            plt.plot(cum_groups.index, cum_groups[i], color=colors[i], alpha=0.8, label=f'Group {i+1}')
            
        # 绘制多空曲线 (使用右轴或单独显著展示)
        cum_ls = (1 + long_short_ret).cumprod()
        plt.plot(cum_ls.index, cum_ls, color='black', linestyle='--', linewidth=2, label='Long-Short')
        
        plt.title(f"Factor Layering: {factor_name}\nIR={metrics.get('IR', 0):.2f}, Ret={metrics.get('Ann. Ret', 0):.1%}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Net Value")
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def calculate_metrics_muti(daily_ret_series, freq_name='Daily'):
        """
        计算绩效指标 (输入序列为已扣费的日收益)
        """
        if len(daily_ret_series) < 10:
            return {'Ann. Ret': 0, 'Sharpe': 0, 'Max DD': 0, 'Win Rate': 0, 'IR': 0}
            
        cum_ret = (1 + daily_ret_series).cumprod()
        total_ret = cum_ret.iloc[-1] - 1
        days = len(daily_ret_series)
        
        # 年化收益 (复利, A股242天)
        ann_ret = (total_ret + 1) ** (242 / days) - 1 
        # 年化波动率
        ann_vol = daily_ret_series.std() * np.sqrt(242)
        # 夏普
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        # IR (简化版: 收益/波动, 实际上如果是多空组合这就是IR)
        ir = sharpe 
        
        # 最大回撤
        cum_max = cum_ret.cummax()
        drawdown = (cum_ret - cum_max) / cum_max
        max_dd = drawdown.min()
        
        return {
            'Frequency': freq_name,
            'Ann. Ret': ann_ret,
            'Ann. Vol': ann_vol,
            'Sharpe': sharpe,
            'IR': ir,
            'Max DD': max_dd
        }

    @staticmethod
    def run_multi_freq_analysis(df_data, factor_col, factor_name=None):
        """
        执行多频率回测 (Weekly, BiWeekly, Monthly) 并输出汇总表
        仅针对 Weekly 频率调用画图
        """
        if factor_name is None: factor_name = factor_col
        print(f"\n{'='*20} Multi-Frequency Analysis: {factor_name} {'='*20}")
        
        # 1. 准备基础数据
        base_df = df_data[['TradingDay', 'SecuCode', factor_col, 'Ret_Hold']].copy()
        base_df = base_df.dropna()
        base_df = base_df.sort_values(['TradingDay', 'SecuCode'])
        
        all_dates = sorted(base_df['TradingDay'].unique())
        results_summary = []
        
        # 2. 循环不同频率
        for freq_name, freq_days in Config.FREQ_DAYS.items():
            # (A) 确定调仓日
            rebalance_dates = all_dates[::freq_days]
            
            # (B) 截面分层 (只在调仓日计算)
            reb_df = base_df[base_df['TradingDay'].isin(rebalance_dates)].copy()
            
            # Rank IC (调仓日均值)
            ic_list = []
            for date, group in reb_df.groupby('TradingDay'):
                if len(group) > 10:
                    ic_list.append(group[factor_col].corr(group['Ret_Hold'], method='spearman'))
            
            # 分组 0-4
            def get_group(x):
                try: return pd.qcut(x, 5, labels=False, duplicates='drop')
                except: return np.nan
            reb_df['Group'] = reb_df.groupby('TradingDay')[factor_col].transform(get_group)
            
            # (C) 映射回日频 (模拟持有)
            sim_df = base_df.merge(reb_df[['TradingDay', 'SecuCode', 'Group']], 
                                 on=['TradingDay', 'SecuCode'], 
                                 how='left')
            sim_df['Group'] = sim_df.groupby('SecuCode')['Group'].ffill(limit=freq_days-1)
            sim_df = sim_df.dropna(subset=['Group'])
            
            # (D) 计算每日各组收益 (等权)
            daily_group_ret = sim_df.groupby(['TradingDay', 'Group'])['Ret_Hold'].mean().unstack()
            
            # (E) 扣除成本 (在调仓日扣除双边成本)
            valid_reb_dates = [d for d in rebalance_dates if d in daily_group_ret.index]
            daily_group_ret.loc[valid_reb_dates] -= Config.COST_RATE_TOTAL
            
            # (F) 多空收益 (Group 4 - Group 0)
            if 4 in daily_group_ret.columns and 0 in daily_group_ret.columns:
                long_short_ret = daily_group_ret[4] - daily_group_ret[0]
            else:
                long_short_ret = pd.Series(0, index=daily_group_ret.index)

            # (G) 记录指标
            metrics = FactorAnalyzer.calculate_metrics_muti(long_short_ret, freq_name)
            metrics['RankIC Mean'] = np.mean(ic_list) if ic_list else 0
            results_summary.append(metrics)
            
            # (H) 仅在 Weekly 频率下画图，且使用指定的画图样式
            if freq_name == 'Weekly':
                FactorAnalyzer.plot_performance_custom(daily_group_ret, long_short_ret, metrics, factor_name)

        # 3. 输出汇总
        df_summary = pd.DataFrame(results_summary)
        cols = ['Frequency', 'RankIC Mean', 'Ann. Ret', 'Sharpe', 'Max DD', 'IR']
        print(f"\n>>> Factor Performance Summary (With Cost): {factor_name}")
        print(df_summary[cols].to_string(float_format="{:.4f}".format))

    @staticmethod
    def plot_performance_custom(group_ret, long_short_ret, metrics, factor_name):
        """
        使用用户指定的样式绘制周频曲线
        """
        plt.figure(figsize=(14, 6))
        
        # 绘制 5 组净值曲线
        cum_groups = (1 + group_ret).cumprod()
        colors = plt.cm.RdYlGn(np.linspace(0, 1, 5)) # 红(差)->绿(好)
        
        for i in range(5):
            # 确保列存在
            if i in cum_groups.columns:
                plt.plot(cum_groups.index, cum_groups[i], color=colors[i], alpha=0.8, label=f'Group {i+1}')
            
        # 绘制多空曲线 (使用右轴或单独显著展示)
        cum_ls = (1 + long_short_ret).cumprod()
        plt.plot(cum_ls.index, cum_ls, color='black', linestyle='--', linewidth=2, label='Long-Short')
        
        plt.title(f"Factor Layering (Weekly): {factor_name}\nIR={metrics.get('IR', 0):.2f}, Ret={metrics.get('Ann. Ret', 0):.1%}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Net Value")
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
    


def plot_factor_correlation(df, method='spearman', title='Factor Correlation Matrix'):
    """
    计算因子相关性并绘制右上角热力图。
    
    参数:
    df     : 包含因子数据的 DataFrame
    method : 'spearman' (推荐，秩相关) 或 'pearson' (线性相关)
    """
    print(f"--- Plotting Correlation Matrix ({method}) ---")
    
    # 1. 筛选因子列 (以 HF_ 开头)
    factor_cols = [c for c in df.columns if c.startswith('HF_')]
    if not factor_cols:
        print("No columns starting with 'HF_' found.")
        return

    # 2. 计算相关性矩阵
    # Spearman 对于因子分析更好，因为它对异常值不敏感，且衡量的是排序关系（Rank IC）
    corr = df[factor_cols].corr(method=method)

    # 3. 生成掩码 (Mask) 用于隐藏左下角
    # np.tril: 返回下三角矩阵 (Triangle Lower)
    # k=0 表示包含对角线，k=-1 表示不包含对角线
    # 这里 mask=True 的部分会被隐藏。我们想隐藏左下角，所以用 tril
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)

    # 4. 绘图
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        corr, 
        mask=mask,            # 应用掩码，只显示右上
        cmap='coolwarm',      # 颜色：红(正相关)-蓝(负相关)
        vmax=1.0, vmin=-1.0,  # 固定色阶范围
        center=0,             # 颜色中心点
        annot=True,           # 显示数值
        fmt='.2f',            # 数值保留两位小数
        square=True,          # 强制单元格为正方形
        linewidths=.5,        # 格子间距
        cbar_kws={"shrink": .5} # 颜色条缩小一点
    )

    plt.title(f"{title} ({method.capitalize()})", fontsize=16)
    plt.xticks(rotation=45, ha='right') # X轴标签旋转45度
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 集成到 Pipeline
# -----------------------------------------------------------------------------
def run_factor_analysis_step(df_final, df_pred):
    """
    在 run_pipeline 中调用此函数，批量分析所有因子
    """
    print(f"\n{'='*60}")
    print(f"Starting Factor Analysis (Individual & ML Composite)")
    print(f"{'='*60}")
    
    # 1. 准备数据
    # 我们需要 'Ret_Hold' (T+1到T+2的收益) 用于真实的因子测试
    # 如果 df_pred 中有 Pred_Score，我们把它合并进 df_final 用作分析
    
    # 为了方便，我们直接构建一个大的 DataFrame 用作分析
    # df_final 包含原始因子 (HF_...)
    # df_pred 包含机器学习预测 (Pred_Score)
    
    # 确保索引对齐
    df_analysis = df_final.copy()
    
    if not df_pred.empty:
        # Merge Prediction Score
        temp_pred = df_pred[['TradingDay', 'SecuCode', 'Pred_Score']].copy()
        df_analysis = pd.merge(df_analysis, temp_pred, on=['TradingDay', 'SecuCode'], how='left')
    
    # 2. 获取所有 HF_ 开头的因子
    factor_cols = [c for c in df_analysis.columns if c.startswith('HF_')]
    
    # 3. 逐个分析原始因子
    for f in factor_cols:
        # 注意：你需要判断因子的方向
        # SmartMoney 通常是越小越好 (S值大意味着散户买入?) -> 视具体定义而定
        # 这里默认 ascending=True (越大越好)，如果发现反了，手动改参数
        
        # 特殊处理 Smart Money (如果是 Barra 定义，通常 S 越大代表散户买入，应该做空)
        is_ascending = True
        if 'SmartMoney' in f:
            is_ascending = False # 假设 SmartMoney 值越大越差
            
        FactorAnalyzer.analyze_factor(df_analysis, f, return_col='Ret_Hold', ascending=is_ascending)

    plot_factor_correlation(df_analysis, method='spearman')

    # 4. 分析机器学习合成因子
    if 'Pred_Score' in df_analysis.columns:
        FactorAnalyzer.analyze_factor(
            df_analysis, 
            'Pred_Score', 
            return_col='Ret_Hold', 
            factor_name='ML_Composite_Factor',
            ascending=True # 预测分越高越好
        )

# -----------------------------------------------------------------------------
# Main Pipeline - 【已修复 Bug 版】
# -----------------------------------------------------------------------------
def run_pipeline(index_path, universe_name):
    print(f"\n{'='*60}")
    print(f"Running Pipeline for: {universe_name} (Fixed: Anti-Leakage & Metrics Crash)")
    print(f"{'='*60}")
    
    # -------------------------------------------------------------------------
    # 1. 数据加载 (Data Loading)
    # -------------------------------------------------------------------------
    try:
        df_univ = DataLoader.load_universe(index_path)
        df_daily = DataLoader.load_daily_quote()
    except Exception as e:
        print(f"Error Loading Data: {e}")
        return

    # -------------------------------------------------------------------------
    # 2. 因子计算 (Factor Engineering)
    # -------------------------------------------------------------------------
    fe = FactorEngine(df_univ, cache_dir=f'./temp_factors_{universe_name}')
    df_fac = fe.run()
    
    if df_fac.empty: 
        print("No factors computed. Exiting.")
        return
    
    del fe
    gc.collect()

    # -------------------------------------------------------------------------
    # 3. 标签构造 (Label Construction for Training)
    # -------------------------------------------------------------------------
    df_tar = DataLoader.construct_label(df_daily)
    
    # -------------------------------------------------------------------------
    # 4. 数据合并与对齐 (Merge & Align) - [Fixed Version]
    # -------------------------------------------------------------------------
    print(">>> [Data] Merging Factors and Labels...")

    # 1. 合并 Factor 和 Universe
    # 这一步通常没有风险
    df_full = pd.merge(
        df_fac, 
        df_univ.rename(columns={'date':'TradingDay', 'symbol':'SecuCode'}), 
        on=['TradingDay', 'SecuCode'], 
        how='inner'
    )
    
    # 2. 合并 Label
    # df_tar 可能已经携带了 Volume, Amount, Shares 等日行情信息
    df_full = pd.merge(
        df_full, 
        df_tar, 
        on=['TradingDay', 'SecuCode'], 
        how='inner'
    )

    # 3. [核心修复] 智能补充 Volume/Shares 数据
    # 只有当 df_full 里缺这些列时，才从 df_daily 里合并进来
    # 防止产生 _x, _y 后缀导致报错
    
    target_cols = ['Volume', 'Amount', 'NonRestrictedShares', 'AFloats']
    
    # 找出 df_full 中还缺失、但 df_daily 中存在的列
    missing_cols = [c for c in target_cols if c not in df_full.columns and c in df_daily.columns]
    
    if missing_cols:
        print(f">>> [Data] Supplementing missing columns: {missing_cols}")
        # 只提取需要的列 + 主键
        df_supplement = df_daily[['TradingDay', 'SecuCode'] + missing_cols].copy()
        
        df_full = pd.merge(
            df_full,
            df_supplement,
            on=['TradingDay', 'SecuCode'],
            how='left'
        )
    else:
        print(">>> [Data] All required columns (Volume/Shares) already exist. Skipping redundant merge.")

    # 4. 清理内存
    del df_fac, df_univ, df_tar
    gc.collect()
    
    # 5. 合并行业与市值 (保持不变)
    df_ind, df_cap = DataLoader.load_industry_cap()
    
    df_full = df_full.sort_values('TradingDay')
    df_full = pd.merge_asof(
        df_full, 
        df_ind[['SecuCode', 'InfoPublDate', 'tag']],
        left_on='TradingDay', 
        right_on='InfoPublDate',
        by='SecuCode', 
        direction='backward'
    )
    
    df_full['Industry'] = df_full['tag'].fillna('Unknown')
    
    # 注意：这里如果 df_full 已经有 LnCap (比如从 feature 里带来的)，也要小心
    # 建议只合并 df_full 里没有的列，或者明确指定 suffix
    cols_cap_merge = ['TradingDay', 'SecuCode', 'LnCap', 'Cap']
    cols_cap_final = [c for c in cols_cap_merge if c not in df_full.columns or c in ['TradingDay', 'SecuCode']]
    
    df_full = pd.merge(
        df_full, 
        df_cap[cols_cap_final], 
        on=['TradingDay', 'SecuCode'], 
        how='inner'
    )
    
    # -------------------------------------------------------------------------
    # 5. 全局特征一致性 (Fix Feature Consistency)
    # -------------------------------------------------------------------------
    print(">>> [Data] 统一行业分类编码...")
    all_industries = sorted(df_full['Industry'].unique().astype(str))
    global_ind_type = pd.CategoricalDtype(categories=all_industries, ordered=False)
    df_full['Industry'] = df_full['Industry'].astype(global_ind_type)
    
    # -------------------------------------------------------------------------
    # 6. 特征工程与模型训练 (ML Pipeline)
    # -------------------------------------------------------------------------
    df_final = process_features_and_label(df_full)
    df_pred = train_model(df_final)

    # =========================================================================
    # 7. 因子分层测试与绘图
    # =========================================================================
    if not df_pred.empty:
        # 注意：这里传入 df_final (含原始因子) 和 df_pred (含 ML 分数)
        run_factor_analysis_step(df_final, df_pred)

    if factor_cols:
        print(f">>> [Analysis] Starting Multi-Frequency Analysis for {len(factor_cols)} factors...")
        analyzer = FactorAnalyzer()
        for fac in factor_cols:
            analyzer.run_multi_freq_analysis(df_final, factor_col=fac)
    else:
        print("Warning: No 'HF_' factors found in df_final.")
    
    # -------------------------------------------------------------------------
    # 8. 回测准备：构建全市场裁判数据
    # -------------------------------------------------------------------------
    # if not df_pred.empty:
    #     print(">>> [Backtest] 构建全市场裁判数据 (防止盲目卖出停牌股)...")
        
    #     # 构建裁判数据 (is_for_backtest=True 保留了停牌日)
    #     df_referee = DataLoader.construct_label(df_daily, is_for_backtest=True)
        
    #     # 【关键修正 1】：必须包含 'Limit_Threshold'，否则回测引擎不知道当日涨跌停限制
    #     cols_needed = ['TradingDay', 'SecuCode', 'Next_PTFlag', 'Next_Open_Chg', 
    #                    'Ret_Hold', 'SettlementDate', 'Limit_Threshold']
        
    #     # 确保列存在 (兼容 DataLoader 是否返回了该列)
    #     existing_cols = [c for c in cols_needed if c in df_referee.columns]
    #     df_referee = df_referee[existing_cols]

    #     # ---------------------------------------------------------------------
    #     # 8. 执行回测 (Backtest Execution)
    #     # ---------------------------------------------------------------------
    #     bt = HedgedBacktest(df_pred, df_referee, universe_name)
    #     res = bt.run()
        
    #     # ---------------------------------------------------------------------
    #     # 9. 结果展示 (Performance Metrics)
    #     # ---------------------------------------------------------------------
    #     if not res.empty:
    #         total_ret = res['Equity'].iloc[-1] - 1
            
    #         # 【关键修正 2】：修复 KeyError
    #         # 原代码使用了 res['Alpha_Net']，但 HedgedBacktest 并未计算该列（因为没有传入 Benchmark）
    #         # 这里改为计算 "策略净收益 (Net Strategy Return)" 的夏普比率
    #         # 如果你是多空策略，这里的 DailyRet 已经隐含了多空对冲；如果是多头策略，这是绝对收益。
            
    #         target_col = 'DailyRet'
            
    #         std_dev = res[target_col].std()
    #         if std_dev == 0 or np.isnan(std_dev):
    #             sharpe = 0
    #         else:
    #             sharpe = res[target_col].mean() / std_dev * np.sqrt(250)
            
    #         cum_max = res['Equity'].cummax()
    #         drawdown = res['Equity'] / cum_max - 1
    #         max_dd = drawdown.min()
            
    #         avg_turnover = res['Turnover'].mean()
    #         avg_hold_count = res['Count'].mean()
            
    #         print(f"\n[{universe_name}] Performance Metrics (Final Fixed):")
    #         print(f"Total Return:    {total_ret:.2%}")
    #         print(f"Sharpe (Net):    {sharpe:.2f}") # 标注明确为 Net Sharpe
    #         print(f"Max Drawdown:    {max_dd:.2%}")
    #         print(f"Avg Daily Turn:  {avg_turnover:.2%}")
    #         print(f"Avg Hold Count:  {avg_hold_count:.1f}")

    #         # 绘图逻辑
    #         if 'SettlementDate' in res.columns:
    #             plot_data = res.dropna(subset=['SettlementDate']).set_index('SettlementDate').sort_index()
    #         else:
    #             plot_data = res.set_index('TradingDay').sort_index()

    #         plt.figure(figsize=(12, 6))
    #         plt.plot(plot_data.index, plot_data['Equity'], label=f"{universe_name} Net Equity")
    #         plt.legend()
    #         plt.title(f"ML Alpha Strategy - {universe_name}\n(Fixed: Limit-20% & Date-Alignment)")
    #         plt.xlabel("Settlement Date")
    #         plt.ylabel("Net Equity")
    #         plt.grid(True)
    #         plt.show()
    #     else:
    #         print("Backtest returned empty results.")
    # else:
    #     print("Prediction DataFrame is empty. Pipeline finished early.")

if __name__ == "__main__":
    if not os.path.exists(Config.ROOT_DIR):
        print(f"Error: Data directory '{Config.ROOT_DIR}' not found.")
    else:
        # run_pipeline(Config.PATH_INDEX_WT_300, "CSI300")
        run_pipeline(Config.PATH_INDEX_WT_500, "CSI500")
        # plt.legend()
        # plt.title("ML Alpha Strategy")
        # plt.show()