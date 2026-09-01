import hashlib
from html import escape
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium


DATA_FILE = Path(__file__).with_name("hc_store.csv")
SERVICE_COLUMNS = ["廁所", "ATM", "座位區"]
REQUIRED_COLUMNS = {
    "公司",
    "店名",
    "地址",
    "電話",
    "經度",
    "緯度",
    *SERVICE_COLUMNS,
}
DEFAULT_CENTER = (24.8016432, 120.9716955)
PRESET_LOCATIONS = {
    "新竹車站": DEFAULT_CENTER,
    "清大夜市": (24.79846, 120.99753),
    "新竹巨城": (24.81000, 120.97750),
    "新竹遠百": (24.80193, 120.96488),
    "新竹廟口": (24.80449, 120.96588),
}
COMPANY_COLORS = {
    "7-11": "#007749",
    "全家": "#1876BD",
    "Hi-Life": "#E31E24",
    "OKmart": "#E60012",
}
SERVICE_ICONS = {"廁所": "🚻", "ATM": "🏧", "座位區": "🪑"}


st.set_page_config(
    page_title="新竹超商服務地圖",
    page_icon="🗺️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_stores(csv_path: str, modified_time: float) -> pd.DataFrame:
    """讀取、驗證並清理超商 CSV；modified_time 用來讓檔案更新後重載。"""
    del modified_time
    data = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"電話": "string"})

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing_text = "、".join(sorted(missing_columns))
        raise ValueError(f"CSV 缺少必要欄位：{missing_text}")

    data = data[list(REQUIRED_COLUMNS)].copy()
    for column in ["公司", "店名", "地址", "電話"]:
        data[column] = data[column].fillna("").astype(str).str.strip()

    data["經度"] = pd.to_numeric(data["經度"], errors="coerce")
    data["緯度"] = pd.to_numeric(data["緯度"], errors="coerce")
    data = data.dropna(subset=["經度", "緯度"])
    data = data[
        data["經度"].between(119, 123)
        & data["緯度"].between(21, 26)
        & data["公司"].isin(["7-11", "全家", "Hi-Life", "OKmart"])
    ].copy()

    for column in SERVICE_COLUMNS:
        data[column] = (
            pd.to_numeric(data[column], errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype(int)
        )

    return data.reset_index(drop=True)


def add_distances(data: pd.DataFrame, center: tuple[float, float]) -> pd.DataFrame:
    """以 Haversine 公式一次計算各門市與中心點的球面距離。"""
    result = data.copy()
    center_latitude, center_longitude = np.radians(center)
    latitudes = np.radians(result["緯度"].to_numpy())
    longitudes = np.radians(result["經度"].to_numpy())
    latitude_delta = latitudes - center_latitude
    longitude_delta = longitudes - center_longitude
    haversine_value = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(center_latitude)
        * np.cos(latitudes)
        * np.sin(longitude_delta / 2) ** 2
    )
    result["距離（公里）"] = 6371.0088 * 2 * np.arcsin(
        np.sqrt(np.clip(haversine_value, 0, 1))
    )
    return result


def store_icon(company: str) -> folium.DivIcon:
    """以接近四家品牌配色的 HTML 標記顯示門市。"""
    if company == "7-11":
        marker_html = """
        <div style="position:relative;width:42px;text-align:center;">
          <div style="background:#fff;border:2px solid #007749;border-radius:6px;
              box-shadow:0 2px 6px rgba(0,0,0,.35);overflow:hidden;
              font-family:Arial,sans-serif;font-weight:900;line-height:1;">
            <div style="height:4px;background:#f58220;"></div>
            <div style="height:4px;background:#e2231a;"></div>
            <div style="padding:4px 1px 5px;color:#007749;font-size:12px;">7-11</div>
          </div>
          <div style="width:0;height:0;margin:-1px auto 0;border-left:7px solid transparent;
              border-right:7px solid transparent;border-top:9px solid #007749;"></div>
        </div>
        """
    elif company == "全家":
        marker_html = """
        <div style="position:relative;width:48px;text-align:center;">
          <div style="background:#fff;border:2px solid #1876bd;border-radius:6px;
              box-shadow:0 2px 6px rgba(0,0,0,.35);overflow:hidden;
              font-family:Arial,sans-serif;font-weight:800;line-height:1;">
            <div style="height:5px;background:#00a651;"></div>
            <div style="padding:5px 1px 6px;color:#1876bd;font-size:11px;">Family</div>
          </div>
          <div style="width:0;height:0;margin:-1px auto 0;border-left:7px solid transparent;
              border-right:7px solid transparent;border-top:9px solid #1876bd;"></div>
        </div>
        """
    elif company == "Hi-Life":
        marker_html = """
        <div style="position:relative;width:48px;text-align:center;">
          <div style="background:#fff;border:2px solid #005baa;border-radius:6px;
              box-shadow:0 2px 6px rgba(0,0,0,.35);overflow:hidden;
              font-family:Arial,sans-serif;font-weight:900;line-height:1;">
            <div style="height:5px;background:#e31e24;"></div>
            <div style="height:3px;background:#ffc72c;"></div>
            <div style="padding:4px 1px 5px;font-size:10px;">
              <span style="color:#e31e24;">Hi</span><span style="color:#005baa;">-Life</span>
            </div>
          </div>
          <div style="width:0;height:0;margin:-1px auto 0;border-left:7px solid transparent;
              border-right:7px solid transparent;border-top:9px solid #005baa;"></div>
        </div>
        """
    else:
        marker_html = """
        <div style="position:relative;width:48px;text-align:center;">
          <div style="background:#fff;border:2px solid #00529b;border-radius:6px;
              box-shadow:0 2px 6px rgba(0,0,0,.35);overflow:hidden;
              font-family:Arial,sans-serif;font-weight:900;line-height:1;">
            <div style="height:4px;background:#00529b;"></div>
            <div style="padding:5px 4px 6px;display:flex;align-items:baseline;
                justify-content:center;gap:1px;white-space:nowrap;">
              <span style="color:#e60012;font-size:12px;">OK</span>
              <span style="color:#00529b;font-size:9px;">mart</span>
            </div>
            <div style="height:3px;background:#e60012;"></div>
          </div>
          <div style="width:0;height:0;margin:-1px auto 0;border-left:7px solid transparent;
              border-right:7px solid transparent;border-top:9px solid #00529b;"></div>
        </div>
        """

    return folium.DivIcon(
        html=marker_html,
        icon_size=(48, 43),
        icon_anchor=(24, 43),
        popup_anchor=(0, -43),
    )


def service_badge(service: str, value: int) -> str:
    """產生門市資訊視窗中的服務徽章。"""
    if value:
        background, color, state = "#E7F5EE", "#076A41", "有"
    else:
        background, color, state = "#F1F3F5", "#6B7280", "無"
    return (
        f'<span style="display:inline-block;margin:2px;padding:3px 7px;'
        f'border-radius:10px;background:{background};color:{color};">'
        f"{SERVICE_ICONS[service]} {escape(service)}：{state}</span>"
    )


def build_map(
    stores: pd.DataFrame,
    center: tuple[float, float],
    center_label: str,
    radius_km: float,
    heatmap_services: list[str],
) -> folium.Map:
    """依目前篩選結果建立可切換門市及服務熱區的 Folium 地圖。"""
    store_map = folium.Map(
        location=list(center),
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="商家和服務熱區統計",
        overlay=False,
        control=True,
        show=True,
    ).add_to(store_map)

    folium.Circle(
        location=list(center),
        radius=radius_km * 1000,
        color="#007749",
        weight=2,
        opacity=0.75,
        dash_array="6, 5",
        fill=True,
        fill_color="#F58220",
        fill_opacity=0.07,
        tooltip=f"{center_label}周邊 {radius_km:g} 公里範圍",
    ).add_to(store_map)

    clusters: dict[str, MarkerCluster] = {}
    for company in ("7-11", "全家", "Hi-Life", "OKmart"):
        company_count = int((stores["公司"] == company).sum())
        if company_count:
            clusters[company] = MarkerCluster(
                name=f"{company} 門市（{company_count} 家）",
                options={
                    "maxClusterRadius": 40,
                    "disableClusteringAtZoom": 16,
                    "spiderfyOnMaxZoom": True,
                },
            ).add_to(store_map)

    for _, store in stores.iterrows():
        company = store["公司"]
        services_html = "".join(
            service_badge(service, int(store[service]))
            for service in SERVICE_COLUMNS
        )
        heading_color = COMPANY_COLORS[company]
        popup_html = f"""
        <div style="font-family:'Microsoft JhengHei',sans-serif;
            min-width:290px;line-height:1.55;">
          <h4 style="margin:6px 0;color:{heading_color};">
            {escape(company)} {escape(store['店名'])}
          </h4>
          <p style="margin:0;"><b>地址：</b>{escape(store['地址'])}</p>
          <p style="margin:0;"><b>電話：</b>{escape(store['電話'])}</p>
          <p style="margin:0 0 5px;"><b>距離：</b>{store['距離（公里）']:.2f} 公里</p>
          <div style="margin-top:5px;">{services_html}</div>
        </div>
        """
        folium.Marker(
            location=[store["緯度"], store["經度"]],
            popup=folium.Popup(popup_html, max_width=390),
            tooltip=f"{company} {store['店名']}｜{store['距離（公里）']:.2f} 公里",
            icon=store_icon(company),
        ).add_to(clusters[company])

    heat_colors = {
        "廁所": ["#E8F5E9", "#8BC34A", "#007749", "#F58220", "#E2231A"],
        "ATM": ["#FFF8E1", "#FFD54F", "#F57F17", "#D84315"],
        "座位區": ["#E0F7FA", "#4DD0E1", "#00838F", "#004D40"],
    }
    for service in heatmap_services:
        heat_data = stores.loc[stores[service] == 1, ["緯度", "經度"]].values.tolist()
        if heat_data:
            HeatMap(
                heat_data,
                name=f"{SERVICE_ICONS[service]} {service}熱區（{len(heat_data)} 家）",
                radius=18,
                blur=12,
                min_opacity=0.35,
                gradient={
                    index / (len(heat_colors[service]) - 1): color
                    for index, color in enumerate(heat_colors[service])
                },
                show=True,
            ).add_to(store_map)

    folium.Marker(
        location=list(center),
        popup=(
            f"{escape(center_label)}："
            f"{center[0]:.7f}, {center[1]:.7f}"
        ),
        tooltip=center_label,
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(store_map)
    folium.LayerControl(collapsed=False).add_to(store_map)
    return store_map


st.markdown(
    """
    <style>
      :root { --brand-green:#007749; --brand-orange:#F58220; --family-blue:#1876BD;
        --hilife-red:#E31E24; --hilife-yellow:#FFC72C; }
      .block-container { padding-top:1.45rem; padding-bottom:2rem; }
      [data-testid="stSidebar"] { border-right:1px solid rgba(128,128,128,.18); }
      .brand-rail { height:7px; display:grid; grid-template-columns:3fr 1fr 3fr 3fr 2fr;
        border-radius:999px; overflow:hidden; margin-bottom:1rem; }
      .brand-rail span:nth-child(1) { background:#007749; }
      .brand-rail span:nth-child(2) { background:#F58220; }
      .brand-rail span:nth-child(3) { background:linear-gradient(90deg,#00A651,#1876BD); }
      .brand-rail span:nth-child(4) {
        background:linear-gradient(90deg,#E31E24 0 58%,#FFC72C 58% 70%,#005BAA 70%); }
      .brand-rail span:nth-child(5) {
        background:linear-gradient(90deg,#E60012 0 50%,#00529B 50%); }
      .hero-title { font-size:clamp(1.8rem,3vw,2.55rem); font-weight:800;
        letter-spacing:-.03em; margin:0; }
      .hero-copy { color:var(--text-color); opacity:.68; margin:.35rem 0 1.1rem; }
      [data-testid="stMetric"] { border-top:3px solid rgba(0,119,73,.7);
        padding:.8rem .9rem; border-radius:0 0 10px 10px;
        background:rgba(128,128,128,.055); }
    </style>
    <div class="brand-rail"><span></span><span></span><span></span><span></span><span></span></div>
    <h1 class="hero-title">新竹超商服務地圖</h1>
    <p class="hero-copy">在同一張地圖比較 7-11、全家、Hi-Life 與 OKmart 門市，並依廁所、ATM、座位區篩選。</p>
    """,
    unsafe_allow_html=True,
)

if not DATA_FILE.exists():
    st.error(f"找不到資料檔：{DATA_FILE.name}")
    st.stop()

try:
    all_stores = load_stores(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
    st.error(f"無法讀取 {DATA_FILE.name}：{error}")
    st.stop()

with st.sidebar:
    st.header("🔎 篩選條件")
    selected_company = st.selectbox(
        "公司", ["全部", "7-11", "全家", "Hi-Life", "OKmart"]
    )
    radius_km = st.slider(
        "中心半徑（公里）",
        min_value=0.5,
        max_value=30.0,
        value=2.0,
        step=0.5,
    )

    with st.expander("📍 調整查詢中心"):
        selected_location = st.selectbox(
            "預設地點",
            [*PRESET_LOCATIONS, "自訂座標"],
        )
        if selected_location == "自訂座標":
            center_latitude = st.number_input(
                "緯度", min_value=21.0, max_value=26.0,
                value=DEFAULT_CENTER[0], format="%.7f"
            )
            center_longitude = st.number_input(
                "經度", min_value=119.0, max_value=123.0,
                value=DEFAULT_CENTER[1], format="%.7f"
            )
            center_label = "自訂查詢中心"
        else:
            center_latitude, center_longitude = PRESET_LOCATIONS[selected_location]
            center_label = selected_location
            st.caption(
                f"座標：{center_latitude:.5f}, {center_longitude:.5f}"
            )

    required_services = st.multiselect(
        "依服務篩選門市",
        SERVICE_COLUMNS,
        placeholder="例如：廁所、ATM",
        help=(
            "勾選「廁所」即可只查看有 WC 的門市；"
            "所選服務也會自動顯示熱區。"
        ),
    )
    service_match_mode = st.radio(
        "多項服務的符合方式",
        ["符合任一項", "同時符合全部"],
        index=0,
        help=(
            "建議使用「符合任一項」查看多種服務；"
            "「同時符合全部」只保留每個勾選項目都有的門市。"
        ),
    )
    heatmap_services = st.multiselect(
        "顯示服務熱區",
        SERVICE_COLUMNS,
        default=["廁所"],
        help="熱區會跟著公司、距離及必備服務一起篩選。",
    )
    st.caption("地圖右上角圖層按鈕可個別開關門市與熱區。")

center = (center_latitude, center_longitude)
filtered_stores = add_distances(all_stores, center)
if selected_company != "全部":
    filtered_stores = filtered_stores[filtered_stores["公司"] == selected_company]
filtered_stores = filtered_stores[filtered_stores["距離（公里）"] <= radius_km]
if required_services:
    service_matches = filtered_stores[required_services].eq(1)
    if service_match_mode == "同時符合全部":
        filtered_stores = filtered_stores[service_matches.all(axis=1)]
    else:
        filtered_stores = filtered_stores[service_matches.any(axis=1)]
filtered_stores = filtered_stores.sort_values("距離（公里）").reset_index(drop=True)

# 「必備服務」不只篩選門市，也自動打開對應熱區，避免操作上看不出變化。
active_heatmap_services = list(
    dict.fromkeys([*heatmap_services, *required_services])
)
empty_heatmap_services = [
    service
    for service in active_heatmap_services
    if int(filtered_stores[service].sum()) == 0
]

metric_columns = st.columns(6)
metric_columns[0].metric("符合門市", f"{len(filtered_stores)} 家")
metric_columns[1].metric("7-11", f"{int((filtered_stores['公司'] == '7-11').sum())} 家")
metric_columns[2].metric("全家", f"{int((filtered_stores['公司'] == '全家').sum())} 家")
metric_columns[3].metric("Hi-Life", f"{int((filtered_stores['公司'] == 'Hi-Life').sum())} 家")
metric_columns[4].metric("OKmart", f"{int((filtered_stores['公司'] == 'OKmart').sum())} 家")
metric_columns[5].metric("有廁所", f"{int(filtered_stores['廁所'].sum())} 家")

if filtered_stores.empty:
    if required_services and service_match_mode == "同時符合全部":
        st.warning(
            "目前範圍沒有同時具備所有勾選服務的門市。"
            "請改選「符合任一項」、放寬距離或減少服務項目。"
        )
    else:
        st.warning("目前條件找不到門市，請放寬距離或減少服務項目。")
elif empty_heatmap_services:
    empty_text = "、".join(empty_heatmap_services)
    st.info(f"目前篩選範圍內沒有「{empty_text}」資料，因此不建立該熱區圖層。")

# 每組篩選條件使用不同元件 key，確保切換各服務熱區時完整刷新地圖。
map_state = "|".join(
    [
        selected_company,
        f"{center[0]:.7f}",
        f"{center[1]:.7f}",
        f"{radius_km:.1f}",
        center_label,
        service_match_mode,
        ",".join(required_services),
        ",".join(active_heatmap_services),
        str(len(filtered_stores)),
    ]
)
map_key = "store_map_" + hashlib.sha1(map_state.encode("utf-8")).hexdigest()[:12]
store_map = build_map(
    filtered_stores,
    center,
    center_label,
    radius_km,
    active_heatmap_services,
)
st_folium(
    store_map,
    key=map_key,
    width=1200,
    height=680,
    returned_objects=[],
)

with st.expander(f"📋 查看篩選結果（{len(filtered_stores)} 家）"):
    display_columns = [
        "公司", "店名", "地址", "電話", "距離（公里）", *SERVICE_COLUMNS
    ]
    display_data = filtered_stores[display_columns].copy()
    display_data["距離（公里）"] = display_data["距離（公里）"].round(2)
    st.dataframe(display_data, width="stretch", hide_index=True)

st.caption(
    "資料說明：服務欄位 1 代表官方門市資料標示有該服務，0 代表未標示；"
    "全家與 Hi-Life 的 ATM 欄位依官方服務標記判定。"
    "Hi-Life 官方門市頁未提供廁所與座位區標記，因此這兩欄記為 0。"
    "OKmart 新竹忠孝店的官方服務篩選標示有 ATM 外幣，因此 ATM 記為 1；"
    "其餘未標示的服務記為 0。"
)
