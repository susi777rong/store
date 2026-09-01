import csv
from html import escape
from pathlib import Path

import folium
from folium.plugins import HeatMap, MarkerCluster
from geopy.distance import geodesic


DATA_FILE = Path(__file__).with_name("hc_store.csv")
REQUIRED_COLUMNS = {
    "公司",
    "店名",
    "地址",
    "電話",
    "經度",
    "緯度",
    "廁所",
    "ATM",
    "座位區",
}
SERVICE_COLUMNS = ["廁所", "ATM", "座位區"]


def load_stores(csv_path: Path) -> list[dict]:
    """讀取7-11、全家、Hi-Life與OKmart的合併CSV資料。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到門市資料檔：{csv_path}")

    stores = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV沒有欄位名稱")

        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing_text = "、".join(sorted(missing_columns))
            raise ValueError(f"CSV缺少必要欄位：{missing_text}")

        for row_number, row in enumerate(reader, start=2):
            try:
                longitude = float(row["經度"])
                latitude = float(row["緯度"])
            except (TypeError, ValueError) as error:
                print(f"-> 略過第{row_number}列：經緯度格式錯誤（{error}）")
                continue

            company = row["公司"].strip()
            if company not in {"7-11", "全家", "Hi-Life", "OKmart"}:
                print(f"-> 略過第{row_number}列：無法辨識公司「{company}」")
                continue

            stores.append(
                {
                    "company": company,
                    "name": row["店名"].strip(),
                    "address": row["地址"].strip(),
                    "phone": row["電話"].strip(),
                    "coords": (latitude, longitude),
                    "services": {
                        service: 1
                        if str(row.get(service, "0")).strip() == "1"
                        else 0
                        for service in SERVICE_COLUMNS
                    },
                }
            )

    return stores


def store_icon(company: str) -> folium.DivIcon:
    """依公司產生四家超商品牌風格的門市圖示。"""
    if company == "7-11":
        marker_html = """
        <div style="position:relative;width:42px;text-align:center;">
          <div style="
              background:#fff;border:2px solid #007749;border-radius:6px;
              box-shadow:0 2px 6px rgba(0,0,0,.35);overflow:hidden;
              font-family:Arial,sans-serif;font-weight:900;line-height:1;">
            <div style="height:4px;background:#f58220;"></div>
            <div style="height:4px;background:#e2231a;"></div>
            <div style="padding:4px 1px 5px;color:#007749;font-size:12px;">7-11</div>
          </div>
          <div style="
              width:0;height:0;margin:-1px auto 0;
              border-left:7px solid transparent;border-right:7px solid transparent;
              border-top:9px solid #007749;"></div>
        </div>
        """
    elif company == "全家":
        marker_html = """
        <div style="position:relative;width:48px;text-align:center;">
          <div style="
              background:#fff;border:2px solid #1876bd;border-radius:6px;
              box-shadow:0 2px 6px rgba(0,0,0,.35);overflow:hidden;
              font-family:Arial,sans-serif;font-weight:800;line-height:1;">
            <div style="height:5px;background:#00a651;"></div>
            <div style="padding:5px 1px 6px;color:#1876bd;font-size:11px;">Family</div>
          </div>
          <div style="
              width:0;height:0;margin:-1px auto 0;
              border-left:7px solid transparent;border-right:7px solid transparent;
              border-top:9px solid #1876bd;"></div>
        </div>
        """
    elif company == "Hi-Life":
        marker_html = """
        <div style="position:relative;width:48px;text-align:center;">
          <div style="
              background:#fff;border:2px solid #005baa;border-radius:6px;
              box-shadow:0 2px 6px rgba(0,0,0,.35);overflow:hidden;
              font-family:Arial,sans-serif;font-weight:900;line-height:1;">
            <div style="height:5px;background:#e31e24;"></div>
            <div style="height:3px;background:#ffc72c;"></div>
            <div style="padding:4px 1px 5px;font-size:10px;">
              <span style="color:#e31e24;">Hi</span><span style="color:#005baa;">-Life</span>
            </div>
          </div>
          <div style="
              width:0;height:0;margin:-1px auto 0;
              border-left:7px solid transparent;border-right:7px solid transparent;
              border-top:9px solid #005baa;"></div>
        </div>
        """
    else:
        marker_html = """
        <div style="position:relative;width:48px;text-align:center;">
          <div style="
              background:#fff;border:2px solid #00529b;border-radius:6px;
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
          <div style="
              width:0;height:0;margin:-1px auto 0;
              border-left:7px solid transparent;border-right:7px solid transparent;
              border-top:9px solid #00529b;"></div>
        </div>
        """

    return folium.DivIcon(
        html=marker_html,
        icon_size=(48, 43),
        icon_anchor=(24, 43),
        popup_anchor=(0, -43),
    )


def service_badge(service: str, value: int) -> str:
    """將服務的1／0狀態顯示成彩色標籤。"""
    background = "#e8f5e9" if value else "#f1f3f5"
    color = "#126b3a" if value else "#757575"
    return (
        f'<span style="display:inline-block;margin:2px;padding:3px 7px;'
        f'border-radius:10px;background:{background};color:{color};">'
        f"{service}：{value}</span>"
    )


def choose_company() -> str:
    """讓使用者選擇顯示全部或四家超商品牌。"""
    print("\n請選擇要顯示的門市：")
    print("  0：全部")
    print("  1：7-11")
    print("  2：全家")
    print("  3：Hi-Life")
    print("  4：OKmart")
    user_choice = input("請輸入選項[直接按Enter使用全部]：").strip()

    aliases = {
        "": "全部",
        "0": "全部",
        "全部": "全部",
        "1": "7-11",
        "7-11": "7-11",
        "711": "7-11",
        "2": "全家",
        "全家": "全家",
        "3": "Hi-Life",
        "Hi-Life": "Hi-Life",
        "hilife": "Hi-Life",
        "萊爾富": "Hi-Life",
        "4": "OKmart",
        "OKmart": "OKmart",
        "okmart": "OKmart",
        "OK": "OKmart",
    }
    selected_company = aliases.get(user_choice, "全部")
    if user_choice not in aliases:
        print(f"-> 無法辨識「{user_choice}」，改為顯示全部")
    return selected_company


# 1. 載入合併CSV並選擇公司
try:
    all_stores = load_stores(DATA_FILE)
except (FileNotFoundError, ValueError) as error:
    raise SystemExit(error) from error

company_counts = {
    company: sum(store["company"] == company for store in all_stores)
    for company in ("7-11", "全家", "Hi-Life", "OKmart")
}
print(
    f"-> 已載入{len(all_stores)}間門市："
    f"7-11有{company_counts['7-11']}間、全家有{company_counts['全家']}間、"
    f"Hi-Life有{company_counts['Hi-Life']}間、OKmart有{company_counts['OKmart']}間"
)
selected_company = choose_company()
company_stores = [
    store
    for store in all_stores
    if selected_company == "全部" or store["company"] == selected_company
]
print(f"-> 本次選擇：{selected_company}，共{len(company_stores)}間")


# 2. 設定中心點，預設為台鐵新竹火車站
default_center = (24.8016432, 120.9716955)
user_center_input = input(
    "請輸入中心點座標（緯度, 經度）[直接按Enter使用新竹火車站]："
).strip()

if not user_center_input:
    center_coords = default_center
    print(f"-> 使用新竹火車站：{center_coords}")
else:
    try:
        coordinate_parts = user_center_input.replace(",", " ").split()
        if len(coordinate_parts) != 2:
            raise ValueError("必須輸入緯度與經度兩個數值")
        center_coords = (float(coordinate_parts[0]), float(coordinate_parts[1]))
        print(f"-> 成功設定中心點：{center_coords}")
    except ValueError as error:
        center_coords = default_center
        print(f"-> 座標格式錯誤（{error}），改用新竹火車站")

try:
    target_radius_km = float(
        input("請輸入搜尋半徑（公里）[直接按Enter使用2公里]：").strip() or 2
    )
    if target_radius_km <= 0:
        raise ValueError("半徑必須大於0")
except ValueError as error:
    target_radius_km = 2.0
    print(f"-> 半徑格式錯誤（{error}），改用{target_radius_km}公里")


# 3. 計算距離並保留半徑內門市
nearby_stores = []
for store in company_stores:
    distance = geodesic(center_coords, store["coords"]).km
    if distance <= target_radius_km:
        nearby_stores.append({**store, "distance": distance})

nearby_stores.sort(key=lambda store: store["distance"])
service_heat_data = {
    service: [
        list(store["coords"])
        for store in nearby_stores
        if store["services"][service] == 1
    ]
    for service in SERVICE_COLUMNS
}


# 4. 建立地圖與搜尋半徑
store_map = folium.Map(
    location=list(center_coords),
    zoom_start=14,
    tiles="OpenStreetMap",
    control_scale=True,
)

folium.Circle(
    location=list(center_coords),
    radius=target_radius_km * 1000,
    color="#007749",
    weight=2,
    opacity=0.75,
    dash_array="6, 5",
    fill=True,
    fill_color="#f58220",
    fill_opacity=0.08,
    tooltip=f"中心點{target_radius_km:g}公里範圍",
).add_to(store_map)


# 5. 四家超商分成獨立圖層，可在右上角分別開關
company_clusters = {}
for company in ("7-11", "全家", "Hi-Life", "OKmart"):
    company_store_count = sum(
        store["company"] == company for store in nearby_stores
    )
    if company_store_count == 0:
        continue

    company_clusters[company] = MarkerCluster(
        name=f"{company}門市（{company_store_count}家）",
        options={
            "maxClusterRadius": 40,
            "disableClusteringAtZoom": 16,
            "spiderfyOnMaxZoom": True,
        },
    ).add_to(store_map)

for store in nearby_stores:
    services_html = "".join(
        service_badge(service, store["services"][service])
        for service in SERVICE_COLUMNS
    )
    heading_color = {
        "7-11": "#007749",
        "全家": "#1876bd",
        "Hi-Life": "#e31e24",
        "OKmart": "#e60012",
    }[store["company"]]
    popup_html = f"""
    <div style="
        font-family:'Microsoft JhengHei',sans-serif;
        min-width:290px;line-height:1.55;">
      <h4 style="margin:6px 0;color:{heading_color};">
        {escape(store['company'])} {escape(store['name'])}門市
      </h4>
      <p style="margin:0;"><b>公司：</b>{escape(store['company'])}</p>
      <p style="margin:0;"><b>地址：</b>{escape(store['address'])}</p>
      <p style="margin:0;"><b>電話：</b>{escape(store['phone'])}</p>
      <p style="margin:0 0 5px;"><b>距中心：</b>{store['distance']:.2f}公里</p>
      <div style="margin-top:5px;">{services_html}</div>
    </div>
    """

    folium.Marker(
        location=list(store["coords"]),
        popup=folium.Popup(popup_html, max_width=390),
        tooltip=(
            f"{store['company']} {store['name']}門市｜"
            f"{store['distance']:.2f}公里"
        ),
        icon=store_icon(store["company"]),
    ).add_to(company_clusters[store["company"]])


# 6. 四項服務各自建立HeatMap圖層
for service in SERVICE_COLUMNS:
    heat_points = service_heat_data[service]
    if heat_points:
        HeatMap(
            heat_points,
            name=f"有{service}門市熱區（{len(heat_points)}家）",
            radius=18,
            blur=12,
            min_opacity=0.35,
            show=service == "廁所",
        ).add_to(store_map)


# 7. 標記中心點並加入圖層控制
folium.Marker(
    location=list(center_coords),
    popup=f"設定中心點：{center_coords}",
    tooltip="地圖中心點",
    icon=folium.Icon(color="red", icon="train", prefix="fa"),
).add_to(store_map)

folium.LayerControl(collapsed=False).add_to(store_map)


# 8. 儲存HTML地圖
output_filename = "hc_store_map.html"
store_map.save(output_filename)
print(f"-> 搜尋範圍內共有{len(nearby_stores)}間門市")
for company in ("7-11", "全家", "Hi-Life", "OKmart"):
    count = sum(store["company"] == company for store in nearby_stores)
    print(f"-> {company}：{count}間")
for service in SERVICE_COLUMNS:
    print(f"-> 有{service}：{len(service_heat_data[service])}間")
print(f"地圖已成功產生：{output_filename}")
