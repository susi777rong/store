import re
from datetime import date, timedelta
from html import escape

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sqlalchemy import create_engine, inspect


HOST = "localhost"
USERNAME = "root"
PASSWORD = "3378621"
PORT = "3306"
DATABASE_NAME = "yt_ticks"
WARMUP_DAYS = 60

matplotlib.rc("font", family="Microsoft JhengHei")
matplotlib.rc("axes", unicode_minus=False)


@st.cache_resource
def get_engine():
    return create_engine(
        f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE_NAME}",
        pool_pre_ping=True,
    )


def normalize_stock_id(stock_id):
    return stock_id.strip().upper()


def is_valid_stock_id(stock_id):
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9.=_^-]{0,29}", stock_id))


def find_stock_table(engine, stock_id):
    """尋找股票資料表，並相容 MySQL 表名大小寫差異。"""
    return next(
        (
            table_name
            for table_name in inspect(engine).get_table_names()
            if table_name.lower() == stock_id.lower()
        ),
        None,
    )


def parse_dates(values):
    """相容舊資料庫的 yy-mm-dd 字串與日期型別。"""
    values = pd.Series(values)
    parsed = pd.to_datetime(values, format="%y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed


def load_db_data(engine, table_name):
    data = pd.read_sql_table(table_name, con=engine)
    date_column = next(
        (column for column in data.columns if column.lower() == "date"), None
    )
    if date_column is None:
        raise ValueError(f"資料表 {table_name} 缺少 Date 欄位。")

    data[date_column] = parse_dates(data[date_column]).to_numpy()
    data = data.dropna(subset=[date_column]).set_index(date_column)
    data.index.name = "Date"
    data = data[~data.index.duplicated(keep="last")].sort_index()
    return data


def last_required_business_day(end_date):
    """週末不要求 DB 必須存在資料，往前取最近的工作日。"""
    required_date = min(end_date, date.today())
    while required_date.weekday() >= 5:
        required_date -= timedelta(days=1)
    return required_date


def db_covers_range(data, start_date, end_date):
    if data.empty:
        return False
    db_start = data.index.min().date()
    db_end = data.index.max().date()
    return db_start <= start_date and db_end >= last_required_business_day(end_date)


def download_stock_data(stock_id, start_date, end_date):
    """下載資料；yfinance 的 end 不包含當日，所以結束日加一天。"""
    download_start = start_date - timedelta(days=WARMUP_DAYS)
    download_end = min(end_date, date.today()) + timedelta(days=1)
    data = yf.download(
        stock_id,
        start=download_start,
        end=download_end,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(data.columns):
        return pd.DataFrame()

    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.index.name = "Date"
    return data


def save_merged_data(engine, table_name, downloaded, existing=None):
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, downloaded], axis=0)
    else:
        combined = downloaded.copy()

    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    db_data = combined.copy()
    db_data.index = db_data.index.strftime("%y-%m-%d")
    db_data.index.name = "Date"
    db_data.to_sql(table_name, engine, index=True, if_exists="replace")


def calculate_yahoo_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_indicators(data):
    data = data.copy()
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=numeric_columns)

    data["SMA_5"] = data["Close"].rolling(window=5).mean()
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["middle_band"] = data["SMA_20"]
    data["std_dev"] = data["Close"].rolling(window=20).std()
    data["upper_band"] = data["middle_band"] + data["std_dev"] * 2
    data["lower_band"] = data["middle_band"] - data["std_dev"] * 2

    low_min = data["Low"].rolling(window=9).min()
    high_max = data["High"].rolling(window=9).max()
    price_range = (high_max - low_min).replace(0, np.nan)
    data["RSV"] = ((data["Close"] - low_min) / price_range) * 100
    data["K"] = data["RSV"].ewm(alpha=1 / 3, adjust=False).mean()
    data["D"] = data["K"].ewm(alpha=1 / 3, adjust=False).mean()
    data["J"] = 3 * data["D"] - 2 * data["K"]

    direction_volume = np.where(
        data["Close"] > data["Close"].shift(1),
        data["Volume"],
        np.where(
            data["Close"] < data["Close"].shift(1), -data["Volume"], 0
        ),
    )
    data["OBV"] = pd.Series(direction_volume, index=data.index).cumsum()
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["DIF"] = data["EMA12"] - data["EMA26"]
    data["MACD"] = data["DIF"].ewm(span=9, adjust=False).mean()
    data["MACD Histogram"] = data["DIF"] - data["MACD"]
    data["RSI5"] = calculate_yahoo_rsi(data["Close"], 5)
    data["RSI10"] = calculate_yahoo_rsi(data["Close"], 10)
    data["BIAS10"] = ((data["Close"] - data["SMA_10"]) / data["SMA_10"]) * 100
    data["BIAS20"] = ((data["Close"] - data["SMA_20"]) / data["SMA_20"]) * 100
    data["B10-B20"] = data["BIAS10"] - data["BIAS20"]
    return data


def analyze_six_indicators(data):
    """以最新交易日資料產生六大指標紅綠燈與文字說明。"""
    latest = data.iloc[-1]
    previous = data.iloc[-6] if len(data) >= 6 else data.iloc[0]
    rows = []
    explanations = []

    light_map = {
        "正向": "🔴 正向",
        "負向": "🟢 負向",
        "持平": "🟡 持平",
    }

    def add_result(indicator, signal, reading, explanation):
        rows.append(
            {
                "指標": indicator,
                "紅綠燈": light_map[signal],
                "目前判讀": reading,
            }
        )
        explanations.append(
            {
                "indicator": indicator,
                "signal": signal,
                "explanation": explanation,
            }
        )

    trend_values = latest[["Close", "SMA_5", "SMA_10", "SMA_20"]]
    if trend_values.isna().any():
        trend_signal = "持平"
        trend_reading = "均線資料不足"
    elif latest["Close"] > latest["SMA_20"] and latest["SMA_5"] > latest["SMA_10"]:
        trend_signal = "正向"
        trend_reading = "股價站上月線，短均線偏多"
    elif latest["Close"] < latest["SMA_20"] and latest["SMA_5"] < latest["SMA_10"]:
        trend_signal = "負向"
        trend_reading = "股價跌破月線，短均線偏空"
    else:
        trend_signal = "持平"
        trend_reading = "股價與均線方向不一致"
    add_result(
        "趨勢／布林",
        trend_signal,
        trend_reading,
        f"收盤 {latest['Close']:.2f}、SMA5 {latest['SMA_5']:.2f}、"
        f"SMA10 {latest['SMA_10']:.2f}、SMA20 {latest['SMA_20']:.2f}。"
        "收盤高於月線且 SMA5 高於 SMA10 判為正向；反向排列判為負向。",
    )

    if pd.isna(latest["OBV"]) or pd.isna(previous["OBV"]):
        obv_signal = "持平"
        obv_reading = "OBV 資料不足"
    elif latest["OBV"] > previous["OBV"]:
        obv_signal = "正向"
        obv_reading = "近五期 OBV 上升，量能偏多"
    elif latest["OBV"] < previous["OBV"]:
        obv_signal = "負向"
        obv_reading = "近五期 OBV 下降，量能偏空"
    else:
        obv_signal = "持平"
        obv_reading = "近五期 OBV 變化不大"
    add_result(
        "OBV 量價",
        obv_signal,
        obv_reading,
        f"目前 OBV {latest['OBV']:,.0f}，五期前 {previous['OBV']:,.0f}。"
        "目前值高於五期前判為正向，低於五期前判為負向。",
    )

    if pd.isna(latest["K"]) or pd.isna(latest["D"]):
        kd_signal = "持平"
        kd_reading = "KD 資料不足"
    elif latest["K"] > latest["D"]:
        kd_signal = "正向"
        kd_reading = "K 線位於 D 線上方"
    elif latest["K"] < latest["D"]:
        kd_signal = "負向"
        kd_reading = "K 線位於 D 線下方"
    else:
        kd_signal = "持平"
        kd_reading = "K、D 線接近"
    kd_zone = "；位於超買區" if latest["K"] >= 80 else "；位於超賣區" if latest["K"] <= 20 else ""
    add_result(
        "KD",
        kd_signal,
        kd_reading + kd_zone,
        f"K={latest['K']:.2f}、D={latest['D']:.2f}、J={latest['J']:.2f}。"
        "K 高於 D 判為正向，K 低於 D 判為負向；80 以上或 20 以下另提示極端區域。",
    )

    macd_values = latest[["DIF", "MACD", "MACD Histogram"]]
    if macd_values.isna().any():
        macd_signal = "持平"
        macd_reading = "MACD 資料不足"
    elif latest["DIF"] > latest["MACD"] and latest["MACD Histogram"] > 0:
        macd_signal = "正向"
        macd_reading = "DIF 高於訊號線，柱體為正"
    elif latest["DIF"] < latest["MACD"] and latest["MACD Histogram"] < 0:
        macd_signal = "負向"
        macd_reading = "DIF 低於訊號線，柱體為負"
    else:
        macd_signal = "持平"
        macd_reading = "線形與柱體訊號不一致"
    add_result(
        "MACD",
        macd_signal,
        macd_reading,
        f"DIF={latest['DIF']:.2f}、MACD={latest['MACD']:.2f}、"
        f"柱體={latest['MACD Histogram']:.2f}。DIF 與柱體同向時形成明確訊號。",
    )

    if pd.isna(latest["RSI5"]):
        rsi_signal = "持平"
        rsi_reading = "RSI 資料不足"
    elif latest["RSI5"] >= 55:
        rsi_signal = "正向"
        rsi_reading = "RSI5 高於 55，動能偏強"
    elif latest["RSI5"] <= 45:
        rsi_signal = "負向"
        rsi_reading = "RSI5 低於 45，動能偏弱"
    else:
        rsi_signal = "持平"
        rsi_reading = "RSI5 位於中性區"
    rsi_zone = "；注意超買" if latest["RSI5"] >= 70 else "；注意超賣" if latest["RSI5"] <= 30 else ""
    add_result(
        "RSI",
        rsi_signal,
        rsi_reading + rsi_zone,
        f"RSI5={latest['RSI5']:.2f}、RSI10={latest['RSI10']:.2f}。"
        "RSI5 高於 55 判為正向，低於 45 判為負向，45～55 判為持平。",
    )

    if pd.isna(latest["B10-B20"]):
        bias_signal = "持平"
        bias_reading = "BIAS 資料不足"
    elif latest["B10-B20"] > 0.1:
        bias_signal = "正向"
        bias_reading = "短期乖離強於中期乖離"
    elif latest["B10-B20"] < -0.1:
        bias_signal = "負向"
        bias_reading = "短期乖離弱於中期乖離"
    else:
        bias_signal = "持平"
        bias_reading = "兩組乖離差距有限"
    add_result(
        "BIAS",
        bias_signal,
        bias_reading,
        f"BIAS10={latest['BIAS10']:.2f}%、BIAS20={latest['BIAS20']:.2f}%、"
        f"差值={latest['B10-B20']:.2f}%。差值高於 0.1 判為正向，低於 -0.1 判為負向。",
    )

    result = pd.DataFrame(rows)
    signal_counts = result["紅綠燈"].str.extract(r"(正向|負向|持平)")[0].value_counts()
    positive_count = int(signal_counts.get("正向", 0))
    negative_count = int(signal_counts.get("負向", 0))
    neutral_count = int(signal_counts.get("持平", 0))
    total_score = positive_count - negative_count
    result["合計"] = [
        f"🔴 正向 {positive_count}",
        f"🟢 負向 {negative_count}",
        f"🟡 持平 {neutral_count}",
        f"總分 {total_score:+d}",
        "",
        "",
    ]
    if positive_count >= negative_count + 2:
        overall = "🔴 整體偏多"
    elif negative_count >= positive_count + 2:
        overall = "🟢 整體偏空"
    else:
        overall = "🟡 多空訊號混合"
    summary = (
        f"{overall}｜偏多 {positive_count} 項、偏空 {negative_count} 項、"
        f"中性 {neutral_count} 項，總分 {total_score:+d}。"
    )
    return result, explanations, summary


def render_indicator_table(indicator_table):
    """建立含跨列總分儲存格的六大指標 HTML 表格。"""
    signals = indicator_table["紅綠燈"].str.extract(r"(正向|負向|持平)")[0]
    positive_count = int((signals == "正向").sum())
    negative_count = int((signals == "負向").sum())
    neutral_count = int((signals == "持平").sum())
    total_score = positive_count - negative_count
    total_class = (
        "score-positive"
        if total_score > 0
        else "score-negative"
        if total_score < 0
        else "score-neutral"
    )
    total_text = f"{total_score:+d}" if total_score else "0"
    count_cells = [
        f'<span class="signal-dot dot-positive"></span>正向 {positive_count}',
        f'<span class="signal-dot dot-negative"></span>負向 {negative_count}',
        f'<span class="signal-dot dot-neutral"></span>持平 {neutral_count}',
    ]

    body_rows = []
    for index, row in indicator_table.iterrows():
        if index < 3:
            summary_cell = f'<td class="summary-count">{count_cells[index]}</td>'
        elif index == 3:
            summary_cell = (
                f'<td class="total-score {total_class}" rowspan="3">'
                f'<span class="total-label">總分</span>'
                f'<span class="total-number">{total_text}</span></td>'
            )
        else:
            summary_cell = ""

        signal = signals.iloc[index]
        dot_class = {
            "正向": "dot-positive",
            "負向": "dot-negative",
            "持平": "dot-neutral",
        }[signal]
        body_rows.append(
            "<tr>"
            f'<td>{escape(str(row["指標"]))}</td>'
            f'<td><span class="signal-dot {dot_class}"></span>{escape(signal)}</td>'
            f'<td>{escape(str(row["目前判讀"]))}</td>'
            f"{summary_cell}"
            "</tr>"
        )

    return f"""
    <style>
    .indicator-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 0.75rem;
    }}
    .indicator-table th, .indicator-table td {{
        padding: 0.58rem 0.7rem;
        border-right: 1px solid rgba(49, 51, 63, 0.13);
        border-bottom: 1px solid rgba(49, 51, 63, 0.13);
        text-align: left;
        vertical-align: middle;
    }}
    .indicator-table th {{
        background: rgba(240, 242, 246, 0.7);
        color: #667085;
        font-weight: 500;
    }}
    .indicator-table tr:last-child td {{ border-bottom: 0; }}
    .indicator-table th:last-child, .indicator-table td:last-child {{ border-right: 0; }}
    .signal-dot {{
        display: inline-block;
        width: 0.8rem;
        height: 0.8rem;
        margin-right: 0.35rem;
        border-radius: 50%;
        vertical-align: -0.05rem;
        box-shadow: inset 0 0 2px rgba(0, 0, 0, 0.2);
    }}
    .dot-positive {{ background: #e53935; }}
    .dot-negative {{ background: #20b26b; }}
    .dot-neutral {{ background: #e7b93f; }}
    .summary-count {{ white-space: nowrap; }}
    .total-score {{
        min-width: 9rem;
        text-align: center !important;
        background: rgba(240, 242, 246, 0.35);
    }}
    .total-label {{
        display: block;
        margin-bottom: 0.25rem;
        color: #667085;
        font-size: 0.95rem;
    }}
    .total-number {{
        display: block;
        font-size: 2.8rem;
        line-height: 1;
        font-weight: 800;
    }}
    .score-positive .total-number {{ color: #e53935; }}
    .score-negative .total-number {{ color: #20a35a; }}
    .score-neutral .total-number {{ color: #d5a514; }}
    </style>
    <table class="indicator-table">
        <thead>
            <tr>
                <th>六大指標</th>
                <th>紅綠燈</th>
                <th>目前判讀</th>
                <th>合計</th>
            </tr>
        </thead>
        <tbody>{''.join(body_rows)}</tbody>
    </table>
    """


def render_indicator_cards(indicator_table, indicator_explanations):
    """建立 3×2 六大指標卡片，並回傳綜合統計。"""
    icons = {
        "趨勢／布林": "📈",
        "OBV 量價": "📊",
        "KD": "🎯",
        "MACD": "〽️",
        "RSI": "⚡",
        "BIAS": "↔️",
    }
    display_names = {
        "趨勢／布林": "均線＋布林通道",
        "OBV 量價": "成交量＋OBV",
        "KD": "KDJ",
    }
    signal_labels = {
        "正向": ("🔴", "偏多", "signal-bull"),
        "負向": ("🟢", "偏空", "signal-bear"),
        "持平": ("🟡", "中性", "signal-neutral"),
    }
    explanation_map = {
        item["indicator"]: item["explanation"].split("。")[0]
        for item in indicator_explanations
    }
    signals = indicator_table["紅綠燈"].str.extract(r"(正向|負向|持平)")[0]
    positive_count = int((signals == "正向").sum())
    negative_count = int((signals == "負向").sum())
    neutral_count = int((signals == "持平").sum())
    total_score = positive_count - negative_count

    if total_score >= 2:
        overall_icon, overall_text = "🔴", "整體偏多"
    elif total_score <= -2:
        overall_icon, overall_text = "🟢", "整體偏空"
    else:
        overall_icon, overall_text = "🟡", "多空訊號混合"

    cards = []
    for index, row in indicator_table.iterrows():
        signal = signals.iloc[index]
        light, label, css_class = signal_labels[signal]
        indicator = str(row["指標"])
        display_name = display_names.get(indicator, indicator)
        reading = str(row["目前判讀"])
        values = explanation_map.get(indicator, "")
        detail = f"{reading}。{values}。" if values else f"{reading}。"
        # 注意：此處 HTML 去除所有前置空白縮排，避免觸發 Markdown 程式碼區塊解析
        cards.append(
            f'<article class="indicator-card">'
            f'<div class="indicator-name">{icons.get(indicator, "📌")} {escape(display_name)}</div>'
            f'<div class="indicator-signal {css_class}">{light} {label}</div>'
            f'<div class="indicator-detail">{escape(detail)}</div>'
            f'</article>'
        )

    dashboard_html = f"""<style>
.indicator-card-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin: 0.4rem 0 1.25rem;
}}
.indicator-card {{
    min-height: 7.4rem;
    padding: 1rem 1.15rem;
    border: 1px solid rgba(49, 51, 63, 0.20);
    border-radius: 12px;
    background: rgba(128, 128, 128, 0.065);
    box-sizing: border-box;
}}
.indicator-name {{
    color: #667085;
    font-size: 0.88rem;
    margin-bottom: 0.4rem;
}}
.indicator-signal {{
    font-size: 1.18rem;
    font-weight: 800;
    margin-bottom: 0.48rem;
}}
.signal-bull {{ color: #ef476f; }}
.signal-bear {{ color: #20b978; }}
.signal-neutral {{ color: #d6ad32; }}
.indicator-detail {{
    color: #596170;
    font-size: 0.88rem;
    line-height: 1.55;
}}
@media (max-width: 900px) {{
    .indicator-card-grid {{ grid-template-columns: 1fr; }}
    .indicator-card {{ min-height: auto; }}
}}
</style><div class="indicator-card-grid">{''.join(cards)}</div>"""

    stats = {
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
        "score": total_score,
        "overall_icon": overall_icon,
        "overall_text": overall_text,
    }
    return dashboard_html, stats


def render_total_score(total_score):
    """建立依正負分顯示紅、綠、黃的大型合計分數。"""
    score_class = (
        "total-bull"
        if total_score > 0
        else "total-bear"
        if total_score < 0
        else "total-neutral"
    )
    score_text = f"{total_score:+d}" if total_score else "0"
    return f"""
    <style>
    .score-panel {{ padding: 0.15rem 0.4rem 0.5rem; }}
    .score-caption {{ color: #667085; font-size: 0.92rem; margin-bottom: 0.15rem; }}
    .score-number {{ font-size: 2.45rem; line-height: 1.15; font-weight: 500; }}
    .total-bull {{ color: #c13d3d; }}
    .total-bear {{ color: #188a54; }}
    .total-neutral {{ color: #b88b17; }}
    </style>
    <div class="score-panel">
        <div class="score-caption">🧮 合計分數 ⓘ</div>
        <div class="score-number {score_class}">{score_text} 分</div>
    </div>
    """


def create_stock_figure(data, stock_id):
    positions = np.arange(len(data))
    tick_step = max(1, len(data) // 12)
    tick_positions = positions[::tick_step]
    tick_labels = [value.strftime("%y-%m-%d") for value in data.index[::tick_step]]

    fig = plt.figure(figsize=(12, 13), layout="constrained")
    ax1 = fig.add_subplot(8, 1, (1, 3))
    mpf.candlestick2_ochl(
        ax1,
        data["Open"],
        data["Close"],
        data["High"],
        data["Low"],
        width=0.8,
        colorup="r",
        colordown="g",
        alpha=1,
    )
    ax1.plot(positions, data["SMA_5"], label="5日均線", color="cyan", lw=0.8)
    ax1.plot(positions, data["SMA_10"], label="10日均線", color="purple", lw=0.8)
    ax1.plot(positions, data["SMA_20"], label="20日均線", color="orange", lw=0.8)
    ax1.plot(positions, data["upper_band"], label="布林上軌", color="g", ls=":")
    ax1.plot(positions, data["lower_band"], label="布林下軌", color="g", ls=":")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([])
    ax1.legend(loc=0)
    ax1.set_title(f"{stock_id} 技術分析")

    up_down_colors = np.select(
        [data["Close"] > data["Close"].shift(1), data["Close"] < data["Close"].shift(1)],
        ["r", "g"],
        default="gray",
    )
    ax2 = fig.add_subplot(8, 1, 4)
    ax2.plot(positions, data["OBV"], color="purple", ls="--", label="OBV")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([])
    ax2.legend(loc=1)
    ax2_volume = ax2.twinx()
    ax2_volume.bar(positions, data["Volume"], color=up_down_colors, width=0.8, alpha=0.8)
    ax2_volume.legend(
        handles=[
            mpatches.Patch(color="red", label="上漲"),
            mpatches.Patch(color="green", label="下跌"),
            mpatches.Patch(color="gray", label="持平"),
        ],
        loc=2,
        title="交易量",
    )

    ax3 = fig.add_subplot(8, 1, 5)
    ax3.plot(positions, data["K"], label="K", color="cyan", lw=0.8)
    ax3.plot(positions, data["D"], label="D", color="purple", lw=0.8)
    ax3.plot(positions, data["J"], label="J", color="orange", ls="--")
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([])
    ax3.legend(loc=0)

    ax4 = fig.add_subplot(8, 1, 6)
    ax4.plot(positions, data["DIF"], label="DIF", color="purple")
    ax4.plot(positions, data["MACD"], label="MACD", color="skyblue")
    macd_colors = np.where(data["MACD Histogram"] >= 0, "r", "g")
    ax4.bar(positions, data["MACD Histogram"], color=macd_colors, alpha=0.8)
    ax4.axhline(0, color="gray", ls="--", lw=1.2)
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels([])
    ax4.legend(loc=2, fontsize=8)

    ax5 = fig.add_subplot(8, 1, 7)
    ax5.plot(positions, data["RSI5"], label="RSI5", color="cyan", lw=0.8)
    ax5.plot(positions, data["RSI10"], label="RSI10", color="purple", lw=0.8)
    ax5.axhline(70, color="red", ls="--", lw=0.8, alpha=0.5)
    ax5.axhline(30, color="green", ls="--", lw=0.8, alpha=0.5)
    ax5.set_ylim(0, 100)
    ax5.set_xticks(tick_positions)
    ax5.set_xticklabels([])
    ax5.legend(loc=2)

    ax6 = fig.add_subplot(8, 1, 8)
    ax6.plot(positions, data["BIAS10"], label="BIAS10", color="cyan")
    ax6.plot(positions, data["BIAS20"], label="BIAS20", color="purple")
    bias_colors = np.where(data["B10-B20"] >= 0, "r", "g")
    ax6.bar(positions, data["B10-B20"], color=bias_colors, alpha=0.8)
    ax6.axhline(0, color="gray", ls="--", lw=1.2)
    bias_values = data["B10-B20"].dropna()
    if not bias_values.empty:
        ax6.set_ylim(min(bias_values.min(), -15) * 1.1, max(bias_values.max(), 15) * 1.1)
    ax6.set_xticks(tick_positions)
    ax6.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax6.legend(loc=2, fontsize=8)
    return fig


def main():
    st.set_page_config(
        page_title="股市整合地端DB分析版", page_icon="📈", layout="wide"
    )

    default_end = date.today()
    default_start = default_end - timedelta(days=180)
    with st.sidebar:
        st.header("查詢條件")
        st.caption("請設定股票代號與分析日期範圍。")
        st.caption("優先讀取本地 MySQL；資料不足時才從 Yahoo Finance 更新。")
    with st.sidebar.form("stock_query"):
        stock_id_input = st.text_input(
            "股票代號",
            value="2330.TW",
            help="台股範例：2330.TW；美股範例：AAPL；指數範例：^TWII",
        )
        start_date = st.date_input("開始日期", value=default_start)
        end_date = st.date_input("結束日期", value=default_end)
        submitted = st.form_submit_button("載入並產生圖表", type="primary")

    stock_id = normalize_stock_id(stock_id_input)
    title_stock_id = stock_id if stock_id else "股票"
    st.title(f"{title_stock_id} 股市整合地端DB分析版")

    if not submitted:
        st.sidebar.info("請輸入股票代號與日期範圍，再按下「載入並產生圖表」。")
        return

    if not stock_id or not is_valid_stock_id(stock_id):
        st.sidebar.warning("股票代號格式不正確，請重新輸入，例如 2330.TW、AAPL 或 ^TWII。")
        return
    if start_date > end_date:
        st.sidebar.warning("開始日期不可晚於結束日期。")
        return
    if start_date > date.today():
        st.sidebar.warning("開始日期不可晚於今天。")
        return

    with st.sidebar.status("正在處理資料…", expanded=True) as status:
        try:
            status.write("正在連線本地 MySQL 資料庫…")
            engine = get_engine()
            table_name = find_stock_table(engine, stock_id)
            db_data = pd.DataFrame()

            if table_name is not None:
                status.write(f"找到本地資料表：{table_name}")
                db_data = load_db_data(engine, table_name)
            else:
                table_name = stock_id.lower()
                status.write("本地 DB 尚無此股票資料，準備向 Yahoo Finance 查詢。")

            if db_covers_range(db_data, start_date, end_date):
                source_message = "本地 DB 已涵蓋日期範圍，直接載入資料庫資料。"
                status.write(source_message)
            else:
                status.write("本地資料不足，正在從 Yahoo Finance 下載…")
                downloaded = download_stock_data(stock_id, start_date, end_date)
                if downloaded.empty:
                    status.update(label="查無股票資料", state="error", expanded=True)
                    st.sidebar.warning(
                        f"查無股票代號 {stock_id}，或該日期範圍沒有可用資料，請確認代號後重試。"
                    )
                    return

                status.write(f"下載完成，共 {len(downloaded):,} 筆；正在更新本地 DB…")
                save_merged_data(engine, table_name, downloaded, db_data)
                status.write("DB 更新完成，正在重新載入資料庫資料…")
                db_data = load_db_data(engine, table_name)
                source_message = "Yahoo Finance 下載完成並寫入 DB，圖表使用重新載入的 DB 資料。"

            status.write("正在計算技術指標並產生圖表…")
            indicator_data = calculate_indicators(db_data)
            chart_data = indicator_data.loc[
                (indicator_data.index.date >= start_date)
                & (indicator_data.index.date <= min(end_date, date.today()))
            ].copy()
            if chart_data.empty:
                status.update(label="日期範圍無交易資料", state="error", expanded=True)
                st.sidebar.warning("指定日期範圍內沒有交易資料，請調整日期後重試。")
                return

            figure = create_stock_figure(chart_data, stock_id)
            status.update(label="完成", state="complete", expanded=False)
        except Exception as error:
            status.update(label="執行失敗", state="error", expanded=True)
            st.sidebar.error(f"執行失敗：{error}")
            return

    st.sidebar.success(source_message)
    indicator_table, indicator_explanations, indicator_summary = analyze_six_indicators(
        chart_data
    )
    st.subheader("🚦 六大指標分析參考")
    st.caption(
        f"資料日期：{chart_data.index.max().strftime('%Y-%m-%d')}｜"
        "紅燈＝偏多、綠燈＝偏空、黃燈＝中性"
    )
    cards_html, indicator_stats = render_indicator_cards(
        indicator_table, indicator_explanations
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    summary_column, score_column = st.columns([3, 1], vertical_alignment="center")
    with summary_column:
        st.info(
            f"綜合參考： {indicator_stats['overall_icon']} "
            f"**{indicator_stats['overall_text']}** ｜ "
            f"🔴 偏多 **{indicator_stats['positive']}** 項 ｜ "
            f"🟢 偏空 **{indicator_stats['negative']}** 項 ｜ "
            f"🟡 中性 **{indicator_stats['neutral']}** 項 ｜ 共 **6** 項"
        )
    with score_column:
        st.markdown(render_total_score(indicator_stats["score"]), unsafe_allow_html=True)

    with st.expander("展開查看六大指標分析說明"):
        st.markdown(f"**綜合結果：{indicator_summary}**")
        st.caption("總分計算：每項偏多 +1、偏空 -1、中性 0。")
        for item in indicator_explanations:
            light = {"正向": "🔴", "負向": "🟢", "持平": "🟡"}[item["signal"]]
            signal_label = {"正向": "偏多", "負向": "偏空", "持平": "中性"}[
                item["signal"]
            ]
            st.markdown(
                f"**{light} {item['indicator']}｜{signal_label}**  \n"
                f"{item['explanation']}"
            )
        st.caption("以上為技術指標規則化判讀，僅供教學與研究參考，不構成投資建議。")

    st.pyplot(figure, width="stretch")
    plt.close(figure)

    with st.expander("查看圖表資料"):
        display_columns = [
            "Open", "High", "Low", "Close", "Volume", "SMA_5", "SMA_10",
            "SMA_20", "K", "D", "J", "DIF", "MACD", "RSI5", "RSI10",
            "BIAS10", "BIAS20",
        ]
        st.dataframe(chart_data[display_columns].sort_index(ascending=False))


if __name__ == "__main__":
    main()
