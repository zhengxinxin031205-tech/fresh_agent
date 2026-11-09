"""
外部因素自动补齐模块
====================
智能体主动连接公开数据源,自动获取门店所在地的:
1. 历史天气(气温/降水) —— Open-Meteo 免费API,无需密钥
2. 法定节假日标记        —— 内置中国节假日日历(2024-2027)

商家只需输入城市名,智能体自动把当地影响因素补进数据集参与分析。
"""
import json
import urllib.request
import urllib.parse
from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd


# ──────────────────── 中国法定节假日日历(2024-2027) ────────────────────
# 格式: "YYYY-MM-DD" -> 节日名称
CHINESE_HOLIDAYS = {}

def _build_holidays():
    """构建2024-2027年中国法定节假日表"""
    holidays = {}
    for year in [2024, 2025, 2026, 2027]:
        # 元旦
        holidays[f"{year}-01-01"] = "元旦"
        # 劳动节
        holidays[f"{year}-05-01"] = "劳动节"
        # 国庆节 10.1-10.3(法定3天)
        holidays[f"{year}-10-01"] = "国庆节"
        holidays[f"{year}-10-02"] = "国庆节"
        holidays[f"{year}-10-03"] = "国庆节"
    # 2024年
    holidays.update({
        "2024-02-10": "春节", "2024-02-11": "春节", "2024-02-12": "春节",
        "2024-04-04": "清明节", "2024-04-05": "清明节", "2024-04-06": "清明节",
        "2024-06-10": "端午节",
        "2024-09-17": "中秋节",
    })
    # 2025年
    holidays.update({
        "2025-01-28": "春节", "2025-01-29": "春节", "2025-01-30": "春节",
        "2025-04-04": "清明节",
        "2025-05-31": "端午节",
        "2025-10-06": "中秋节",
    })
    # 2026年(农历日期为推算值)
    holidays.update({
        "2026-02-17": "春节", "2026-02-18": "春节", "2026-02-19": "春节",
        "2026-04-05": "清明节",
        "2026-06-19": "端午节",
        "2026-09-25": "中秋节",
    })
    # 2027年(农历日期为推算值)
    holidays.update({
        "2027-02-06": "春节", "2027-02-07": "春节", "2027-02-08": "春节",
        "2027-04-05": "清明节",
        "2027-06-09": "端午节",
        "2027-09-15": "中秋节",
    })
    return holidays

CHINESE_HOLIDAYS = _build_holidays()


def get_holiday(date_str: str) -> Tuple[int, str]:
    """查询某天是否为法定节假日,返回 (是否节假日0/1, 节日名称)"""
    name = CHINESE_HOLIDAYS.get(date_str, "")
    return (1, name) if name else (0, "")


# ──────────────────── 城市地理编码 ────────────────────
def geocode_city(city_name: str) -> Optional[Tuple[float, float, str]]:
    """
    调用 Open-Meteo 地理编码API,把城市名转为经纬度
    返回 (纬度, 经度, 标准城市名) 或 None
    """
    try:
        query = urllib.parse.quote(city_name)
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=zh&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "FreshAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        return (float(r["latitude"]), float(r["longitude"]), r.get("name", city_name))
    except Exception:
        return None


# ──────────────────── 历史天气获取 ────────────────────
def fetch_weather(lat: float, lon: float,
                  start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    调用 Open-Meteo 历史 weather API,获取指定日期范围逐日天气
    返回 DataFrame: date, temperature(日均温), precipitation(降水量mm)
    """
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&timezone=Asia%2FShanghai"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "FreshAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        daily = data.get("daily", {})
        if not daily or "time" not in daily:
            return None
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temperature": [(a + b) / 2 for a, b in zip(
                daily.get("temperature_2m_max", []),
                daily.get("temperature_2m_min", []))],
            "precipitation": daily.get("precipitation_sum", []),
        })
        return df
    except Exception:
        return None


# ──────────────────── 因素补齐主入口 ────────────────────
def enrich_factors(df: pd.DataFrame, city: str) -> dict:
    """
    智能补齐当地影响因素:
    1. 根据城市名获取经纬度
    2. 根据数据日期范围获取逐日天气(气温/降水)
    3. 根据日期标记法定节假日
    4. 合并到原数据集

    返回 {"ok": bool, "message": str, "added_columns": [...], "city": str}
    """
    if "date" not in df.columns:
        return {"ok": False, "message": "数据中没有日期列,无法补齐当地因素"}

    dmin = df["date"].min()
    dmax = df["date"].max()
    # Open-Meteo 历史API只提供到5天前的数据
    today = date.today()
    max_allowed = today - timedelta(days=5)
    if dmax.date() > max_allowed:
        dmax = pd.Timestamp(max_allowed)
    if dmin.date() >= dmax.date():
        return {"ok": False, "message": "数据日期范围太短或太新,无法获取历史天气"}

    # 1) 地理编码
    geo = geocode_city(city)
    if geo is None:
        return {"ok": False, "message": f"找不到城市「{city}」,请确认城市名(如:北京、上海、广州)"}
    lat, lon, std_name = geo

    # 2) 获取天气
    weather = fetch_weather(lat, lon, dmin.strftime("%Y-%m-%d"), dmax.strftime("%Y-%m-%d"))
    if weather is None or len(weather) == 0:
        return {"ok": False, "message": f"获取「{std_name}」天气数据失败,请稍后重试"}

    # 3) 合并天气
    added = []
    df_enriched = df.copy()
    weather_map = weather.set_index("date")
    date_key = df_enriched["date"].dt.normalize()

    if "temperature" not in df_enriched.columns:
        temps = date_key.map(weather_map["temperature"])
        df_enriched["temperature"] = temps.values
        added.append("temperature")
    if "precipitation" not in df_enriched.columns:
        precs = date_key.map(weather_map["precipitation"])
        df_enriched["precipitation"] = precs.values
        added.append("precipitation")

    # 4) 节假日标记
    if "is_holiday" not in df_enriched.columns:
        hol_results = df_enriched["date"].dt.strftime("%Y-%m-%d").apply(get_holiday)
        df_enriched["is_holiday"] = hol_results.apply(lambda x: x[0])
        df_enriched["holiday_name"] = hol_results.apply(lambda x: x[1])
        added.append("is_holiday")
        added.append("holiday_name")

    if not added:
        return {"ok": True, "message": f"数据中已包含天气和节假日因素,无需补齐",
                "added_columns": [], "city": std_name, "df": df_enriched}

    # 天气缺失值填充(同一日期内前向填充)
    for col in ["temperature", "precipitation"]:
        if col in df_enriched.columns and col in added:
            df_enriched[col] = df_enriched.groupby("date")[col].transform(
                lambda s: s.ffill().bfill())
            df_enriched[col] = df_enriched[col].fillna(
                df_enriched[col].mean() if df_enriched[col].notna().any() else 0)

    n_days = len(weather)
    return {
        "ok": True,
        "message": f"已为「{std_name}」补齐 {n_days} 天当地因素:气温、降水、节假日标记",
        "added_columns": added,
        "city": std_name,
        "weather_days": n_days,
        "df": df_enriched,
    }