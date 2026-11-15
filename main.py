

# ============================================================
# N.A.D.A
# Agricultural Decision Support System for Egypt
# Backend: Python + FastAPI (Vercel Serverless Ready)
# Data Sources:
#   - NASA POWER (Daily + Climatology + SMAP Soil Moisture + Solar/ET)
#   - NASA SRTM Elevation (via Open-Elevation / Open-Meteo Elevation)
#   - ISRIC SoilGrids REST API v2.0 (Hybrid Soil Profile)
#   - Open-Meteo (16-day Forecast + FAO ET0 + Seasonal Outlook)
#   - Open ExchangeRate API (Live USD/EGP + Input Cost Indexing)
#   - Gemini AI Advisor (All World Languages)
# Version: 2.0.0
# ============================================================

# ============================================================
# 1. IMPORTS
# ============================================================

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from statistics import mean
from typing import Optional, List
import logging
import math
import os
import threading
import time

import requests
from requests.adapters import HTTPAdapter
import uvicorn

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("NADA_Agri")

APP_VERSION = "2.0.0"


# ============================================================
# 3. FASTAPI APPLICATION
# ============================================================

APP_VERSION = "2.0.0"
app = FastAPI(
    title="N.A.D.A API",
    description="Agricultural Decision Support System for Egypt",
    version=APP_VERSION
)


# ============================================================
# 4. MIDDLEWARE (CORS & GZIP)
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=500)


# ============================================================
# 5. EXTERNAL API CONFIGURATION, TIMEOUTS & CACHE (VERCEL SAFE)
# ============================================================

# Strict 3.5-second timeout for EVERY external HTTP call to prevent Vercel 504 timeouts
EXTERNAL_TIMEOUT = 3.5

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_CLIMATOLOGY_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

# Live NASA POWER AG Parameters:
# T2M: 2m Temperature (C)
# PRECTOTCORR: Precipitation (mm/day)
# RH2M: Relative Humidity (%)
# WS10M: Wind Speed 10m (m/s)
# ALLSKY_SFC_SW_DWN: Solar Radiation (MJ/m2/day)
# GWETTOP: Surface Soil Wetness 0-5cm (0..1)
# GWETROOT: Root Zone Soil Wetness 0-100cm from SMAP/MERRA-2 (0..1)
# EVPTRNS: Evapotranspiration Energy Flux (W/m2)
NASA_PARAMETERS = "T2M,PRECTOTCORR,RH2M,WS10M,ALLSKY_SFC_SW_DWN,GWETTOP,GWETROOT,EVPTRNS"
NASA_CLIMATOLOGY_PARAMETERS = "T2M,PRECTOTCORR,RH2M,WS10M,ALLSKY_SFC_SW_DWN"
NASA_COMMUNITY = "AG"

WEATHER_PERIOD_DAYS = 30
NASA_DATA_DELAY_DAYS = 7

# NASA SRTM Elevation via Open-Elevation (with Open-Meteo Elevation secondary fallback)
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# ISRIC SoilGrids REST API v2.0
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Open-Meteo Forecast & Seasonal Outlook
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_SEASONAL_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"

FORECAST_DAYS = 16
OUTLOOK_DAYS = 30
OUTLOOK_MIN_DAYS = 14

# Live Exchange Rate & Commodity Indexing
EXCHANGE_RATE_URL = "https://open.er-api.com/v6/latest/USD"
BASELINE_USD_EGP = 48.50
FALLBACK_USD_EGP = 49.50

MONTH_KEYS = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]

DAYS_IN_MONTH = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Cache TTLs (seconds)
NASA_DAILY_CACHE_TTL = 6 * 3600
NASA_CLIMATOLOGY_CACHE_TTL = 24 * 3600
FORECAST_CACHE_TTL = 30 * 60
OUTLOOK_CACHE_TTL = 6 * 3600
SOILGRIDS_CACHE_TTL = 24 * 3600
ELEVATION_CACHE_TTL = 24 * 3600
MARKET_CACHE_TTL = 6 * 3600

# Unit conversions & irrigation saving ratios
HECTARE_TO_FEDDAN = 2.38095
FEDDAN_AREA_M2 = 4200.83

IRRIGATION_EFFICIENCY = {
    "التنقيط": 0.90,
    "الرش": 0.78,
    "الغمر": 0.58,
    "أخرى": 0.65,
}

IRRIGATION_SAVING_VS_FLOOD = {
    "التنقيط": 0.35,
    "الرش": 0.25,
    "الغمر": 0.0,
    "أخرى": 0.0,
}

RICE_AWD_SAVING_VS_FLOOD = 0.15


class _TTLCache:
    """Thread-safe in-memory TTL cache with bounded size."""

    def __init__(self, max_items=512):
        self._store = {}
        self._lock = threading.Lock()
        self._max_items = max_items

    def get(self, key):
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if now >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key, value, ttl_seconds):
        now = time.monotonic()
        with self._lock:
            if len(self._store) >= self._max_items:
                expired_keys = [
                    k for k, (exp, _) in self._store.items() if now >= exp
                ]
                for k in expired_keys:
                    del self._store[k]
                if len(self._store) >= self._max_items:
                    oldest_key = next(iter(self._store))
                    del self._store[oldest_key]
            self._store[key] = (now + ttl_seconds, value)


_DATA_CACHE = _TTLCache(max_items=512)
_THREAD_LOCAL = threading.local()


def _get_http_session():
    """Return a thread-local requests.Session with connection pooling."""
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=12, pool_maxsize=12)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _THREAD_LOCAL.session = session
    return session


def _coord_cache_key(latitude, longitude):
    return (round(float(latitude), 4), round(float(longitude), 4))


# ============================================================
# 6. GOVERNORATES AND APPROXIMATE COORDINATES
# ============================================================

GOVERNORATES = {

    "القاهرة": (30.0444, 31.2357),
    "الجيزة": (30.0131, 31.2089),
    "الإسكندرية": (31.2001, 29.9187),
    "الدقهلية": (31.0364, 31.3807),
    "البحر الأحمر": (27.2579, 33.8116),
    "البحيرة": (30.8481, 30.3436),
    "الفيوم": (29.3084, 30.8428),
    "الغربية": (30.8754, 31.0335),
    "الإسماعيلية": (30.5965, 32.2719),
    "المنوفية": (30.5972, 30.9876),
    "المنيا": (28.1099, 30.7503),
    "القليوبية": (30.3292, 31.2168),
    "الوادي الجديد": (25.4514, 30.5463),
    "السويس": (29.9668, 32.5498),
    "أسوان": (24.0889, 32.8998),
    "أسيوط": (27.1809, 31.1837),
    "بني سويف": (29.0661, 31.0994),
    "بورسعيد": (31.2653, 32.3019),
    "دمياط": (31.4175, 31.8144),
    "جنوب سيناء": (28.1099, 33.5840),
    "كفر الشيخ": (31.1107, 30.9388),
    "مطروح": (31.3543, 27.2373),
    "الأقصر": (25.6872, 32.6396),
    "قنا": (26.1551, 32.7160),
    "شمال سيناء": (31.1321, 33.8033),
    "الشرقية": (30.5877, 31.5020),
    "سوهاج": (26.5569, 31.6948),

}

COASTAL_DELTA_GOVERNORATES = {
    "الإسكندرية", "البحيرة", "كفر الشيخ", "دمياط", "بورسعيد",
}

DESERT_GOVERNORATES = {
    "مطروح", "الوادي الجديد", "جنوب سيناء", "شمال سيناء", "البحر الأحمر",
}

HOT_SUMMER_GOVERNORATES = {
    "أسيوط", "سوهاج", "قنا", "الأقصر", "أسوان", "الوادي الجديد",
}

GOVERNORATE_EN = {
    "القاهرة": "Cairo", "الجيزة": "Giza", "الإسكندرية": "Alexandria",
    "الدقهلية": "Dakahlia", "البحر الأحمر": "Red Sea", "البحيرة": "Beheira",
    "الفيوم": "Fayoum", "الغربية": "Gharbia", "الإسماعيلية": "Ismailia",
    "المنوفية": "Monufia", "المنيا": "Minya", "القليوبية": "Qalyubia",
    "الوادي الجديد": "New Valley", "السويس": "Suez", "أسوان": "Aswan",
    "أسيوط": "Assiut", "بني سويف": "Beni Suef", "بورسعيد": "Port Said",
    "دمياط": "Damietta", "جنوب سيناء": "South Sinai",
    "كفر الشيخ": "Kafr El Sheikh", "مطروح": "Matrouh",
    "الأقصر": "Luxor", "قنا": "Qena", "شمال سيناء": "North Sinai",
    "الشرقية": "Sharqia", "سوهاج": "Sohag",
}

# Approximate baseline elevation (m ASL) by governorate for SRTM fallback
GOVERNORATE_ELEVATION_FALLBACK = {
    "الإسكندرية": 5.0, "بورسعيد": 2.0, "دمياط": 3.0, "كفر الشيخ": 4.0,
    "البحيرة": 6.0, "الدقهلية": 8.0, "الغربية": 10.0, "الشرقية": 12.0,
    "المنوفية": 14.0, "القليوبية": 17.0, "الإسماعيلية": 13.0, "السويس": 14.0,
    "القاهرة": 25.0, "الجيزة": 22.0, "الفيوم": 20.0, "بني سويف": 29.0,
    "المنيا": 39.0, "أسيوط": 52.0, "سوهاج": 61.0, "قنا": 73.0,
    "الأقصر": 82.0, "أسوان": 108.0, "الوادي الجديد": 120.0,
    "مطروح": 35.0, "شمال سيناء": 30.0, "جنوب سيناء": 220.0,
    "البحر الأحمر": 45.0,
}

REGION_RULES = {

    "قصب السكر": {
        "main": ("المنيا", "أسيوط", "سوهاج", "قنا", "الأقصر", "أسوان"),
        "weight": 2,
        "ar_ok": "{gov} من المحافظات المعتادة لزراعة {name}.",
        "en_ok": "{gov} is one of the usual governorates for growing {name}.",
        "ar_bad": (
            "{name} يتركز عادةً في: {regions}. محافظتك ({gov}) خارج هذه "
            "المناطق؛ تأكد من التعاقد مع المصانع والجدوى مع الإرشاد الزراعي."
        ),
        "en_bad": (
            "{name} is usually grown in: {regions}. Your governorate ({gov}) "
            "is outside these areas; confirm sugar-mill contracts and "
            "feasibility with the extension service."
        ),
    },

    "الأرز": {
        "main": ("كفر الشيخ", "الدقهلية", "الشرقية", "الغربية", "البحيرة",
                 "دمياط", "بورسعيد"),
        "weight": 2,
        "ar_ok": "{gov} من محافظات شمال الدلتا المعتادة لزراعة {name}.",
        "en_ok": "{gov} is one of the northern Delta governorates usually used for {name}.",
        "ar_bad": (
            "زراعة {name} في مصر مقيدة بمناطق تحددها وزارة الري، وغالبًا في: "
            "{regions}. محافظتك ({gov}) خارجها؛ تأكد من المساحات المسموح بها."
        ),
        "en_bad": (
            "Growing {name} in Egypt is restricted to zones set by the Ministry "
            "of Irrigation, mostly in: {regions}. Your governorate ({gov}) is "
            "outside them; confirm the permitted areas."
        ),
    },

    "بنجر السكر": {
        "main": ("كفر الشيخ", "الدقهلية", "البحيرة", "الغربية", "الشرقية",
                 "دمياط", "الفيوم", "المنيا"),
        "weight": 1,
        "ar_ok": "{gov} من المحافظات المعتادة لزراعة {name}.",
        "en_ok": "{gov} is one of the usual governorates for growing {name}.",
        "ar_bad": (
            "{name} يُزرع غالبًا في: {regions}. محافظتك ({gov}) ليست من "
            "المناطق المعتادة؛ تأكد من توفر مصانع التسلّم قريبة منك."
        ),
        "en_bad": (
            "{name} is mostly grown in: {regions}. Your governorate ({gov}) is "
            "not a usual area; confirm a processing factory is nearby."
        ),
    },

}


# ============================================================
# 7. APPLICATION OPTIONS
# ============================================================

CROPS = [

    "القمح",
    "الأرز",
    "الذرة",
    "القطن",
    "قصب السكر",
    "بنجر السكر",
    "الطماطم",
    "البطاطس",
    "البصل",
    "الثوم",
    "الفول البلدي",
    "الفاصوليا",
    "العدس",
    "الحمص",
    "الفول السوداني",
    "عباد الشمس",
    "السمسم",
    "البرسيم",
    "العنب",
    "المانجو",
    "الموالح",
    "الموز",
    "الزيتون",
    "الفراولة",
    "مش متأكد",

]


MONTHS = [

    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",

]

MONTHS_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


IRRIGATION_METHODS = [

    "الغمر",
    "التنقيط",
    "الرش",
    "أخرى",

]


WATER_SOURCES = [

    "مياه النيل أو الترع",
    "مياه الآبار",
    "مياه معالجة",
    "مصدر آخر",

]


SOIL_TYPES = [

    "رملية",
    "طينية",
    "طميية",
    "لا أعرف",

]


STANDING_WATER_OPTIONS = [

    "نعم",
    "لا",
    "أحيانًا",

]


AGRICULTURAL_PROBLEMS = [

    "نقص المياه",
    "ملوحة التربة",
    "اصفرار النباتات",
    "الآفات والحشرات",
    "الأمراض النباتية",
    "ضعف نمو المحصول",
    "لا توجد مشكلة",
    "مشكلة أخرى",

]

SUPPORTED_LANGUAGES = ["ar", "en"]

AREA_UNITS = ["فدان", "هكتار"]


# ============================================================
# 8. FARM DATA MODELS
# ============================================================

class FarmData(BaseModel):

    previous_crop: str = "غير معروف"

    # Eight required questions

    governorate: str = Field(min_length=1)

    crop: str = Field(min_length=1)

    planting_month: int = Field(ge=1, le=12)

    irrigation_method: str = Field(min_length=1)

    water_source: str = Field(min_length=1)

    soil_type: str = Field(min_length=1)

    standing_water: str = Field(min_length=1)

    agricultural_problem: str = Field(min_length=1)

    # Optional second crop for side-by-side comparison

    compare_crop: Optional[str] = None

    # Optional information

    farmer_name: Optional[str] = None

    farm_area: Optional[float] = Field(
        default=None,
        gt=0
    )

    farm_area_unit: Optional[str] = None

    soil_photo_url: Optional[str] = None

    crop_photo_url: Optional[str] = None

    soil_lab_results: Optional[dict] = None

    irrigation_water_results: Optional[dict] = None

    additional_notes: Optional[str] = None

    # Optional manual economic overrides (for dynamic calculator flexibility)

    custom_cost_per_feddan: Optional[float] = Field(
        default=None,
        gt=0
    )

    custom_price_per_ton: Optional[float] = Field(
        default=None,
        gt=0
    )

    custom_yield_ton_per_feddan: Optional[float] = Field(
        default=None,
        gt=0
    )

    # Optional GPS coordinates

    latitude: Optional[float] = Field(
        default=None,
        ge=-90,
        le=90
    )

    longitude: Optional[float] = Field(
        default=None,
        ge=-180,
        le=180
    )

    # Optional language of the assessment text (accepts any world language code)

    language: Optional[str] = "ar"


class CompareRequest(FarmData):

    compare_crop: str = Field(min_length=1)

class SoilHealthEvaluation(BaseModel):
    nitrogen_impact: str
    organic_matter_impact: str
    erosion_risk: str

class PestAndFallowManagement(BaseModel):
    fallow_period_days: int
    pest_cycle_broken: bool
    explanation: str

class RotationStrategy(BaseModel):
    strategy_name: str
    strategy_type: str
    sequence: List[str]
    reasoning: str 
    evidence: str
    benefits: List[str]
    gaps: List[str]
    risks: List[str]
    soil_health: SoilHealthEvaluation
    pest_management: PestAndFallowManagement

CROP_FAMILIES = {
    "القمح": "نجيلية", "الأرز": "نجيلية", "الذرة": "نجيلية", "القطن": "خبازية",
    "قصب السكر": "نجيلية", "بنجر السكر": "قطيفية", "الطماطم": "باذنجانية",
    "البطاطس": "باذنجانية", "البصل": "نرجسية", "الثوم": "نرجسية",
    "الفول البلدي": "بقولية", "الفاصوليا": "بقولية", "العدس": "بقولية",
    "الحمص": "بقولية", "الفول السوداني": "بقولية", "عباد الشمس": "نجمية",
    "السمسم": "بيدالية", "البرسيم": "بقولية", "العنب": "كرمية",
    "المانجو": "بطمية", "الموالح": "سذابية", "الموز": "موزية",
    "الزيتون": "زيتونية", "الفراولة": "وردية",
}

def evaluate_soil_health(sequence: List[str]) -> SoilHealthEvaluation:
    families = [CROP_FAMILIES.get(c, "غير معروف") for c in sequence]
    n_impact = "إيجابي جداً: المحاصيل البقولية تثبت النيتروجين الجوي في التربة." if "بقولية" in families else "سلبي: استنزاف مستمر للنيتروجين، يتطلب تسميداً كيميائياً مكثفاً."
    om_impact = "جيد: بقايا الجذور والسيقان تزيد من المادة العضوية." if "نجيلية" in families else "متوسط: الاعتماد على الكومبوست."
    erosion = "منخفض: الغطاء النباتي المستمر يقلل من التعرية." if len(sequence) >= 3 else "مرتفع: فترات بدون غطاء تعرض التربة للتآكل."
    return SoilHealthEvaluation(nitrogen_impact=n_impact, organic_matter_impact=om_impact, erosion_risk=erosion)

def evaluate_pest_and_fallow(
    sequence: List[str],
    fallow_days_override: Optional[int] = None
) -> PestAndFallowManagement:

    families = [CROP_FAMILIES.get(c, "غير معروف") for c in sequence]

    pest_broken = True

    for i in range(len(families) - 1):
        if families[i] == families[i + 1]:
            pest_broken = False
            break

    fallow_days = (
        fallow_days_override
        if fallow_days_override is not None
        else (45 if len(sequence) == 2 else 20)
    )

    exp = (
        "تنوع العائلات يمنع تراكم مسببات الأمراض."
        if pest_broken
        else "زراعة نفس العائلة تباعاً يضاعف خطر الآفات كالنيماتودا."
    )

    if fallow_days_override is not None:
        exp += (
            f" (فترة الحقل الفاضي محسوبة من NDVI "
            f"لقمر MODIS: {fallow_days_override} يوماً)"
        )

    return PestAndFallowManagement(
        fallow_period_days=fallow_days,
        pest_cycle_broken=pest_broken,
        explanation=exp
    )
def calculate_fallow_days_from_ndvi(ndvi_data):
    if not ndvi_data or not ndvi_data.get("available"):
        return None

    series = ndvi_data.get("series", [])
    if not series:
        return None

    low_ndvi_count = 0

    for point in reversed(series):
        value = point.get("ndvi")

        if value is None or value >= 0.2:
            break

        low_ndvi_count += 1

    return low_ndvi_count * 16


def generate_rotation_strategies(
    previous_crop: str,
    target_crop: str,
    fallow_days_override: Optional[int] = None
) -> List[RotationStrategy]:

    strategies = []

    has_prev = previous_crop and previous_crop != "غير معروف"

    seq_1 = ([previous_crop] if has_prev else []) + [target_crop]


    strategies.append(RotationStrategy(
        strategy_name="التدوير الحالي للمزارع",
        strategy_type="Current",
        sequence=seq_1,
        reasoning="هذا هو النمط بناءً على اختيارات المزارع.",
        evidence="البيانات المدخلة.",
        benefits=["سهولة التطبيق لتوفر خبرة المزارع."],
        gaps=["قد يفتقر إلى التنوع البيولوجي."],
        risks=["تراكم الآفات."],
        soil_health=evaluate_soil_health(seq_1),
        pest_management=evaluate_pest_and_fallow(
            seq_1, fallow_days_override=fallow_days_override
        )
    ))

    seq_2 = ([previous_crop] if has_prev else []) + [
        target_crop,
        "البرسيم" if target_crop != "البرسيم" else "القمح"
    ]

    strategies.append(RotationStrategy(
        strategy_name="الدورة المحسنة (صحة التربة)",
        strategy_type="Eco-Friendly",
        sequence=seq_2,
        reasoning="إدخال بقوليات لتعويض النيتروجين وتحسين الخواص.",
        evidence="الدراسات تؤكد زيادة الإنتاجية.",
        benefits=["تثبيت النيتروجين الطبيعي."],
        gaps=["عائد المحصول الثالث قد يكون أقل."],
        risks=["تقلبات الأسعار."],
        soil_health=evaluate_soil_health(seq_2),
        pest_management=evaluate_pest_and_fallow(
            seq_2, fallow_days_override=fallow_days_override
        )
    ))

    break_crop = (
        "القطن"
        if target_crop in ["الطماطم", "البطاطس"]
        else "البطاطس"
    )

    seq_3 = ([previous_crop] if has_prev else []) + [
        break_crop, target_crop
    ]

    strategies.append(RotationStrategy(
        strategy_name="استراتيجية كسر الآفات والعائد المرتفع",
        strategy_type="High-Yield",
        sequence=seq_3,
        reasoning="تغيير جذري للعائلة لكسر الأمراض.",
        evidence="توصيات البحوث الزراعية.",
        benefits=["القضاء على الآفات المتوطنة.", "عائد مادي جيد."],
        gaps=["استهلاك مائي أعلى."],
        risks=["تكلفة مبدئية عالية."],
        soil_health=evaluate_soil_health(seq_3),
        pest_management=evaluate_pest_and_fallow(
            seq_3, fallow_days_override=fallow_days_override
        )
    ))

    return strategies
# ============================================================
# 9. VALIDATE FARM DATA
# ============================================================

def validate_farm_data(farm):

    required_answers = {

        "governorate": farm.governorate,

        "crop": farm.crop,

        "planting_month": farm.planting_month,

        "irrigation_method": farm.irrigation_method,

        "water_source": farm.water_source,

        "soil_type": farm.soil_type,

        "standing_water": farm.standing_water,

        "agricultural_problem": farm.agricultural_problem,

    }

    for question, answer in required_answers.items():

        if answer is None:

            raise HTTPException(
                status_code=422,
                detail={
                    "code": "MISSING_REQUIRED_ANSWER",
                    "question": question,
                    "message": "Please answer this required question."
                }
            )

        if isinstance(answer, str) and not answer.strip():

            raise HTTPException(
                status_code=422,
                detail={
                    "code": "EMPTY_REQUIRED_ANSWER",
                    "question": question,
                    "message": "This question cannot be empty."
                }
            )

    if farm.governorate not in GOVERNORATES:

        raise HTTPException(
            status_code=422,
            detail="Please select a valid Egyptian governorate."
        )

    if farm.crop not in CROPS:

        raise HTTPException(
            status_code=422,
            detail="Please select a valid crop."
        )

    if farm.compare_crop is not None:

        if not farm.compare_crop.strip():

            raise HTTPException(
                status_code=422,
                detail="Second crop for comparison cannot be empty."
            )

        if farm.compare_crop not in CROPS or farm.compare_crop == "مش متأكد":

            raise HTTPException(
                status_code=422,
                detail="Please select a valid specific crop for comparison."
            )

        if farm.crop == farm.compare_crop:

            raise HTTPException(
                status_code=422,
                detail="Please select two different crops to compare."
            )

    if not 1 <= farm.planting_month <= 12:

        raise HTTPException(
            status_code=422,
            detail="Planting month must be between 1 and 12."
        )

    if farm.irrigation_method not in IRRIGATION_METHODS:

        raise HTTPException(
            status_code=422,
            detail="Invalid irrigation method."
        )

    if farm.water_source not in WATER_SOURCES:

        raise HTTPException(
            status_code=422,
            detail="Invalid water source."
        )

    if farm.soil_type not in SOIL_TYPES:

        raise HTTPException(
            status_code=422,
            detail="Invalid soil type."
        )

    if farm.standing_water not in STANDING_WATER_OPTIONS:

        raise HTTPException(
            status_code=422,
            detail="Invalid standing water answer."
        )

    if farm.agricultural_problem not in AGRICULTURAL_PROBLEMS:

        raise HTTPException(
            status_code=422,
            detail="Invalid agricultural problem."
        )

    if (farm.latitude is None) != (farm.longitude is None):

        raise HTTPException(
            status_code=422,
            detail="Both GPS coordinates must be provided together."
        )

    if farm.farm_area is not None:

        if farm.farm_area_unit not in AREA_UNITS:

            raise HTTPException(
                status_code=422,
                detail="Please specify the farm area unit."
            )

    if farm.language is not None and (not isinstance(farm.language, str) or not farm.language.strip()):

        raise HTTPException(
            status_code=422,
            detail="Invalid language code."
        )

    return True


# ============================================================
# 10. OPTIONS ENDPOINT
# ============================================================

@app.get("/api/options")
def get_options():

    problems_en = {
        "نقص المياه": "Water shortage",
        "ملوحة التربة": "Soil salinity",
        "اصفرار النباتات": "Leaf yellowing",
        "الآفات والحشرات": "Pests and insects",
        "الأمراض النباتية": "Plant diseases",
        "ضعف نمو المحصول": "Weak crop growth",
        "لا توجد مشكلة": "No problem",
        "مشكلة أخرى": "Other problem",
    }

    standing_en = {
        "نعم": "Yes",
        "لا": "No",
        "أحيانًا": "Sometimes",
    }

    soil_en = {
        "رملية": "Sandy",
        "طينية": "Clay",
        "طميية": "Loam",
        "لا أعرف": "I don't know",
    }

    irrigation_en = {
        "الغمر": "Flood irrigation",
        "التنقيط": "Drip irrigation",
        "الرش": "Sprinkler irrigation",
        "أخرى": "Other",
    }

    water_en = {
        "مياه النيل أو الترع": "Nile or canal water",
        "مياه الآبار": "Well water",
        "مياه معالجة": "Treated water",
        "مصدر آخر": "Other source",
    }

    crops_en = {crop: prof["en"] for crop, prof in CROP_PROFILES.items()}
    crops_en["مش متأكد"] = "Not sure"

    return {

        "required_questions": [
            "governorate",
            "crop",
            "planting_month",
            "irrigation_method",
            "water_source",
            "soil_type",
            "standing_water",
            "agricultural_problem",
        ],

        "governorates": list(GOVERNORATES.keys()),

        "crops": CROPS,

        "months": [
            {
                "number": index + 1,
                "name": month,
                "name_en": MONTHS_EN[index],
            }
            for index, month in enumerate(MONTHS)
        ],

        "irrigation_methods": IRRIGATION_METHODS,

        "water_sources": WATER_SOURCES,

        "soil_types": SOIL_TYPES,

        "standing_water": STANDING_WATER_OPTIONS,

        "agricultural_problems": AGRICULTURAL_PROBLEMS,

        "labels_en": {
            "governorates": GOVERNORATE_EN,
            "crops": crops_en,
            "irrigation_methods": irrigation_en,
            "water_sources": water_en,
            "soil_types": soil_en,
            "standing_water": standing_en,
            "agricultural_problems": problems_en,
        },

        "optional_information": [
            "compare_crop",
            "farmer_name",
            "farm_area",
            "soil_photo_url",
            "crop_photo_url",
            "soil_lab_results",
            "irrigation_water_results",
            "custom_cost_per_feddan",
            "custom_price_per_ton",
            "custom_yield_ton_per_feddan",
            "additional_notes",
        ],

    }


# ============================================================
# 11. SELECT FARM LOCATION
# ============================================================

def select_farm_location(farm):

    if farm.latitude is not None and farm.longitude is not None:

        return {

            "latitude": farm.latitude,

            "longitude": farm.longitude,

            "coordinate_type": "GPS",

        }

    latitude, longitude = GOVERNORATES[farm.governorate]

    return {

        "latitude": latitude,

        "longitude": longitude,

        "coordinate_type": "Approximate governorate coordinates",

    }


# ============================================================
# 12. PREPARE NASA REQUEST DATES
# ============================================================

def prepare_nasa_dates():

    end_date = date.today() - timedelta(
        days=NASA_DATA_DELAY_DAYS
    )

    start_date = end_date - timedelta(
        days=WEATHER_PERIOD_DAYS - 1
    )

    return start_date, end_date


# ============================================================
# 13. GET NASA DATA (TIMEOUT=3.5s + SMART EGYPT FALLBACK)
# ============================================================

_EGYPT_MONTHLY_NORMALS = {
    1:  {"temp": 14.0, "rain_m": 8.0,  "rh": 65.0, "solar": 13.5, "wind": 3.4},
    2:  {"temp": 15.5, "rain_m": 6.0,  "rh": 60.0, "solar": 16.5, "wind": 3.7},
    3:  {"temp": 18.5, "rain_m": 4.0,  "rh": 55.0, "solar": 20.5, "wind": 4.0},
    4:  {"temp": 22.8, "rain_m": 2.0,  "rh": 48.0, "solar": 24.0, "wind": 4.2},
    5:  {"temp": 26.8, "rain_m": 1.0,  "rh": 44.0, "solar": 26.5, "wind": 4.1},
    6:  {"temp": 29.5, "rain_m": 0.2,  "rh": 45.0, "solar": 28.5, "wind": 4.2},
    7:  {"temp": 30.8, "rain_m": 0.0,  "rh": 50.0, "solar": 28.0, "wind": 4.0},
    8:  {"temp": 30.6, "rain_m": 0.1,  "rh": 53.0, "solar": 26.5, "wind": 3.8},
    9:  {"temp": 28.4, "rain_m": 0.5,  "rh": 55.0, "solar": 23.0, "wind": 3.8},
    10: {"temp": 24.8, "rain_m": 2.0,  "rh": 58.0, "solar": 19.0, "wind": 3.6},
    11: {"temp": 20.0, "rain_m": 5.0,  "rh": 62.0, "solar": 15.0, "wind": 3.4},
    12: {"temp": 15.6, "rain_m": 7.0,  "rh": 66.0, "solar": 12.8, "wind": 3.3},
}


def _build_nasa_daily_fallback(latitude, start_date, end_date):
    """Instant regional fallback for Egypt if NASA POWER exceeds 3.5s on Vercel."""
    lat_offset = max(-3.0, min(4.5, (28.5 - float(latitude)) * 0.55))
    is_coastal = float(latitude) >= 30.9
    t2m, rain, rh, ws, sw, gwettop, gwetroot, evptrns = {}, {}, {}, {}, {}, {}, {}, {}

    cursor = start_date
    while cursor <= end_date:
        key = cursor.strftime("%Y%m%d")
        norm = _EGYPT_MONTHLY_NORMALS[cursor.month]
        t_val = round(norm["temp"] + lat_offset, 2)
        rain_val = round((norm["rain_m"] * (1.8 if is_coastal else 0.7)) / 30.0, 3)
        rh_val = round(min(82.0, norm["rh"] + (10.0 if is_coastal else -6.0)), 2)
        ws_val = round(norm["wind"], 2)
        sw_val = round(norm["solar"], 2)
        et_w_m2 = round(max(25.0, sw_val * 3.8), 2)

        t2m[key] = t_val
        rain[key] = rain_val
        rh[key] = rh_val
        ws[key] = ws_val
        sw[key] = sw_val
        gwettop[key] = 0.38 if is_coastal else 0.28
        gwetroot[key] = 0.44 if is_coastal else 0.34
        evptrns[key] = et_w_m2

        cursor += timedelta(days=1)

    return {
        "T2M": t2m,
        "PRECTOTCORR": rain,
        "RH2M": rh,
        "WS10M": ws,
        "ALLSKY_SFC_SW_DWN": sw,
        "GWETTOP": gwettop,
        "GWETROOT": gwetroot,
        "EVPTRNS": evptrns,
        "_is_fallback": True,
    }


def _build_nasa_climatology_fallback(latitude):
    """Instant regional climatology fallback for all 12 months."""
    lat_offset = max(-3.0, min(4.5, (28.5 - float(latitude)) * 0.55))
    is_coastal = float(latitude) >= 30.9
    result = {}
    for m in range(1, 13):
        norm = _EGYPT_MONTHLY_NORMALS[m]
        result[m] = {
            "temp": round(norm["temp"] + lat_offset, 2),
            "rain_mm_day": round((norm["rain_m"] * (1.8 if is_coastal else 0.7)) / DAYS_IN_MONTH[m - 1], 3),
            "rh": round(min(82.0, norm["rh"] + (10.0 if is_coastal else -6.0)), 2),
            "ws": round(norm["wind"], 2),
            "solar": round(norm["solar"], 2),
        }
    return result


def get_nasa_data(latitude, longitude):

    start_date, end_date = prepare_nasa_dates()

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    parameters = {
        "parameters": NASA_PARAMETERS,
        "community": NASA_COMMUNITY,
        "longitude": longitude,
        "latitude": latitude,
        "start": start_str,
        "end": end_str,
        "format": "JSON",
    }

    cache_key = ("nasa_daily_v2", *_coord_cache_key(latitude, longitude), start_str, end_str)
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached, parameters

    session = _get_http_session()

    try:
        logger.info("Fetching live NASA POWER daily data (timeout=%ss)", EXTERNAL_TIMEOUT)
        response = session.get(
            NASA_URL,
            params=parameters,
            timeout=EXTERNAL_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        weather_data = data["properties"]["parameter"]

        for required in ("T2M", "PRECTOTCORR", "RH2M", "WS10M"):
            if required not in weather_data:
                raise KeyError(required)

        weather_data["_is_fallback"] = False
        _DATA_CACHE.set(cache_key, weather_data, NASA_DAILY_CACHE_TTL)
        return weather_data, parameters

    except (
        requests.exceptions.RequestException,
        ValueError, KeyError, TypeError, AttributeError
    ) as error:
        logger.warning("NASA POWER daily timed out or failed (%s); using regional fallback.", error)
        fallback_data = _build_nasa_daily_fallback(latitude, start_date, end_date)
        return fallback_data, parameters


# ============================================================
# 13B. GET NASA MONTHLY CLIMATOLOGY (TIMEOUT=3.5s + FALLBACK)
# ============================================================

def _climatology_value(value):

    try:
        number = float(value)
    except (ValueError, TypeError):
        return None

    if not math.isfinite(number) or number == -999.0:
        return None

    return number


def get_nasa_climatology(latitude, longitude):

    cache_key = ("nasa_climatology_v2", *_coord_cache_key(latitude, longitude))
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    parameters = {
        "parameters": NASA_CLIMATOLOGY_PARAMETERS,
        "community": NASA_COMMUNITY,
        "longitude": longitude,
        "latitude": latitude,
        "format": "JSON",
    }

    names = (
        ("T2M", "temp"),
        ("PRECTOTCORR", "rain_mm_day"),
        ("RH2M", "rh"),
        ("WS10M", "ws"),
        ("ALLSKY_SFC_SW_DWN", "solar"),
    )

    session = _get_http_session()

    try:
        response = session.get(
            NASA_CLIMATOLOGY_URL,
            params=parameters,
            timeout=EXTERNAL_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()["properties"]["parameter"]

        result = {}
        for index, key in enumerate(MONTH_KEYS, start=1):
            result[index] = {
                short: _climatology_value(data.get(name, {}).get(key))
                for name, short in names
            }

        if all(month["temp"] is None for month in result.values()):
            raise KeyError("no temperature climatology")

        _DATA_CACHE.set(cache_key, result, NASA_CLIMATOLOGY_CACHE_TTL)
        return result

    except (
        requests.exceptions.RequestException,
        ValueError, KeyError, TypeError, AttributeError
    ) as error:
        logger.warning("NASA climatology timed out or failed (%s); using fallback.", error)
        return _build_nasa_climatology_fallback(latitude)


# ============================================================
# 13C. NASA SRTM ELEVATION (OPEN-ELEVATION + FALLBACK)
# ============================================================

def get_elevation_srtm(latitude, longitude, governorate=None, open_meteo_elevation=None):

    cache_key = ("srtm_elevation", *_coord_cache_key(latitude, longitude))
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    session = _get_http_session()
    elevation_m = None
    source = "Open-Elevation (NASA SRTM)"

    try:
        resp = session.get(
            OPEN_ELEVATION_URL,
            params={"locations": f"{latitude},{longitude}"},
            timeout=EXTERNAL_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results") or []
        if results and results[0].get("elevation") is not None:
            elevation_m = float(results[0]["elevation"])
    except (requests.exceptions.RequestException, ValueError, TypeError, KeyError, IndexError) as error:
        logger.warning("Open-Elevation failed (%s); trying fallback.", error)

    if elevation_m is None and open_meteo_elevation is not None:
        try:
            elevation_m = float(open_meteo_elevation)
            source = "Open-Meteo Copernicus/SRTM DEM Fallback"
        except (ValueError, TypeError):
            elevation_m = None

    if elevation_m is None:
        try:
            resp2 = session.get(
                OPEN_METEO_ELEVATION_URL,
                params={"latitude": latitude, "longitude": longitude},
                timeout=EXTERNAL_TIMEOUT,
            )
            resp2.raise_for_status()
            elev_arr = resp2.json().get("elevation") or []
            if elev_arr and elev_arr[0] is not None:
                elevation_m = float(elev_arr[0])
                source = "Open-Meteo DEM Fallback"
        except (requests.exceptions.RequestException, ValueError, TypeError, KeyError, IndexError):
            elevation_m = None

    if elevation_m is None:
        elevation_m = GOVERNORATE_ELEVATION_FALLBACK.get(governorate, 25.0)
        source = "Governorate Elevation Baseline Fallback"

    elevation_m = round(elevation_m, 1)

    if elevation_m <= 15.0:
        terrain_zone = "low_delta"
        flood_efficiency_note_ar = (
            f"الارتفاع المنخفض ({elevation_m} م فوق سطح البحر) يشير لقرب مستوى الماء الأرضي؛ "
            "الري بالغمر هنا يرفع خطر التغدق وتملح التربة مقارنة بالتنقيط."
        )
        flood_efficiency_note_en = (
            f"Low elevation ({elevation_m} m ASL) indicates a shallow water table; "
            "flood irrigation increases waterlogging and salinity risks compared to drip."
        )
        recommended_upgrade = "التنقيط"
    elif elevation_m <= 80.0:
        terrain_zone = "valley_terrace"
        flood_efficiency_note_ar = (
            f"الارتفاع المتوسط ({elevation_m} م فوق سطح البحر) مناسب للزراعة؛ "
            "التحول من الغمر إلى التنقيط أو الرش يقلل الفاقد بالبخر والرشح العميق بنسبة 25–35%."
        )
        flood_efficiency_note_en = (
            f"Moderate elevation ({elevation_m} m ASL) suits farming; "
            "switching from flood to drip/sprinkler cuts evaporation and deep percolation losses by 25-35%."
        )
        recommended_upgrade = "التنقيط"
    else:
        terrain_zone = "high_plateau_desert"
        flood_efficiency_note_ar = (
            f"الأرض مرتفعة ({elevation_m} م فوق سطح البحر) مما يرفع تكلفة رفع وضخ المياه؛ "
            "الري بالغمر غير كفء هنا ويُفضّل بشدة الري المضغوط (التنقيط أو الرش)."
        )
        flood_efficiency_note_en = (
            f"High elevation ({elevation_m} m ASL) increases water pumping head and energy cost; "
            "flood irrigation is inefficient here and pressurized drip/sprinkler is strongly advised."
        )
        recommended_upgrade = "التنقيط"

    result = {
        "elevation_m": elevation_m,
        "source": source,
        "terrain_zone": terrain_zone,
        "recommended_system_by_elevation": recommended_upgrade,
        "flood_vs_drip_note_ar": flood_efficiency_note_ar,
        "flood_vs_drip_note_en": flood_efficiency_note_en,
    }

    if "Fallback" not in source:
        _DATA_CACHE.set(cache_key, result, ELEVATION_CACHE_TTL)

    return result


# ============================================================
# 13D. HYBRID SOIL SYSTEM (ISRIC SOILGRIDS + MANUAL LAB OVERRIDE)
# ============================================================

def _classify_soil_texture_from_percentages(clay_pct, sand_pct):
    """Approximate USDA soil texture mapping into Egypt's 3 core soil types."""
    if clay_pct is None or sand_pct is None:
        return None
    if sand_pct >= 55.0 and clay_pct < 20.0:
        return "رملية"
    if clay_pct >= 35.0:
        return "طينية"
    return "طميية"


def _soilgrids_fallback_by_region(governorate, soil_type):
    if governorate in DESERT_GOVERNORATES or soil_type == "رملية":
        return {"clay_pct": 11.0, "sand_pct": 74.0, "silt_pct": 15.0, "ph": 7.9}
    if governorate in COASTAL_DELTA_GOVERNORATES or soil_type == "طينية":
        return {"clay_pct": 44.0, "sand_pct": 22.0, "silt_pct": 34.0, "ph": 7.8}
    return {"clay_pct": 28.0, "sand_pct": 34.0, "silt_pct": 38.0, "ph": 7.6}


def get_soilgrids_data(latitude, longitude, governorate=None, reported_soil_type=None):

    cache_key = ("soilgrids", *_coord_cache_key(latitude, longitude))
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    session = _get_http_session()
    params = [
        ("lon", longitude),
        ("lat", latitude),
        ("property", "clay"),
        ("property", "sand"),
        ("property", "phh2o"),
        ("depth", "0-5cm"),
        ("depth", "5-15cm"),
        ("value", "mean"),
    ]

    try:
        resp = session.get(
            SOILGRIDS_URL,
            params=params,
            timeout=EXTERNAL_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        layers = data.get("properties", {}).get("layers", [])

        extracted = {}
        for layer in layers:
            prop_name = layer.get("name")
            depths = layer.get("depths") or []
            vals = [
                d.get("values", {}).get("mean")
                for d in depths
                if d.get("values", {}).get("mean") is not None
            ]
            if vals:
                d_factor = layer.get("unit_measure", {}).get("d_factor") or 10
                extracted[prop_name] = round(mean(vals) / float(d_factor), 2)

        clay = extracted.get("clay")
        sand = extracted.get("sand")
        ph = extracted.get("phh2o")

        if clay is None and sand is None and ph is None:
            raise ValueError("Empty SoilGrids properties")

        silt = (
            round(max(0.0, 100.0 - clay - sand), 2)
            if clay is not None and sand is not None else None
        )
        result = {
            "clay_pct": clay,
            "sand_pct": sand,
            "silt_pct": silt,
            "ph": ph,
            "source": "ISRIC SoilGrids REST API v2.0",
            "is_fallback": False,
        }
        _DATA_CACHE.set(cache_key, result, SOILGRIDS_CACHE_TTL)
        return result

    except (
        requests.exceptions.RequestException,
        ValueError, KeyError, TypeError, AttributeError
    ) as error:
        logger.warning("SoilGrids API timed out or failed (%s); using regional fallback.", error)
        fb = _soilgrids_fallback_by_region(governorate, reported_soil_type)
        return {
            "clay_pct": fb["clay_pct"],
            "sand_pct": fb["sand_pct"],
            "silt_pct": fb["silt_pct"],
            "ph": fb["ph"],
            "source": "Regional Soil Baseline Fallback (SoilGrids Unavailable)",
            "is_fallback": True,
        }


def _safe_float(val):
    if val is None or val == "":
        return None
    try:
        num = float(val)
        return num if math.isfinite(num) else None
    except (ValueError, TypeError):
        return None


def build_hybrid_soil_profile(farm, soilgrids_data, lang):
    """
    Hybrid Soil Logic:
    Manual laboratory results (soil_lab_results) ALWAYS take top priority over SoilGrids.
    SoilGrids fills any missing clay/sand/pH metrics when manual lab values are absent.
    """
    lab = farm.soil_lab_results or {}
    manual_ph = _safe_float(lab.get("ph"))
    manual_ec = _safe_float(lab.get("ec"))
    manual_n = _safe_float(lab.get("nitrogen_ppm"))
    manual_clay = _safe_float(lab.get("clay_pct"))
    manual_sand = _safe_float(lab.get("sand_pct"))

    has_any_manual = any(
        v is not None for v in (manual_ph, manual_ec, manual_n, manual_clay, manual_sand)
    )
    has_manual_texture = manual_clay is not None and manual_sand is not None

    final_clay = manual_clay if manual_clay is not None else soilgrids_data.get("clay_pct")
    final_sand = manual_sand if manual_sand is not None else soilgrids_data.get("sand_pct")
    final_silt = (
        round(max(0.0, 100.0 - final_clay - final_sand), 2)
        if final_clay is not None and final_sand is not None else soilgrids_data.get("silt_pct")
    )
    final_ph = manual_ph if manual_ph is not None else soilgrids_data.get("ph")

    inferred_texture = _classify_soil_texture_from_percentages(final_clay, final_sand)

    effective_soil_type = farm.soil_type
    if farm.soil_type == "لا أعرف" and inferred_texture is not None:
        effective_soil_type = inferred_texture
    elif has_manual_texture and inferred_texture is not None:
        effective_soil_type = inferred_texture

    if has_any_manual and (has_manual_texture and manual_ph is not None):
        source_code = "manual_lab"
        source_label = L(lang, "تحليل معملي يدوي (أولوية قصوى)", "Manual Laboratory Analysis (Top Priority)")
    elif has_any_manual:
        source_code = "hybrid_manual_and_soilgrids"
        source_label = L(
            lang,
            "هجين: تحليل معملي يدوي + مسح SoilGrids الفضائي",
            "Hybrid: Manual Lab Results + Satellite/Global SoilGrids"
        )
    elif not soilgrids_data.get("is_fallback", True):
        source_code = "soilgrids_satellite"
        source_label = L(
            lang,
            "مسح فضائي/عالمي تلقائي (ISRIC SoilGrids 250m)",
            "Satellite/Global Gridded Survey (ISRIC SoilGrids 250m)"
        )
    else:
        source_code = "regional_fallback"
        source_label = L(
            lang,
            "تقدير إقليمي احتياطي + وصف المزارع للتربة",
            "Regional Soil Baseline Fallback + Farmer Description"
        )

    return {
        "source_code": source_code,
        "source_label": source_label,
        "clay_percent": final_clay,
        "sand_percent": final_sand,
        "silt_percent": final_silt,
        "ph": final_ph,
        "ec_ds_m": manual_ec,
        "nitrogen_ppm": manual_n,
        "farmer_reported_soil_type": farm.soil_type,
        "inferred_texture_from_data": inferred_texture,
        "effective_soil_type": effective_soil_type,
        "soilgrids_raw": soilgrids_data,
        "manual_lab_override_applied": has_any_manual,
    }


# ============================================================
# 13E. WEATHER FORECAST & FAO ET0 FROM OPEN-METEO (TIMEOUT=3.5s)
# ============================================================

def _open_meteo_get(url, params):

    session = _get_http_session()

    try:
        response = session.get(
            url,
            params=params,
            timeout=EXTERNAL_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, ValueError) as error:
        logger.warning("Open-Meteo request failed (%s): %s", url, error)
        return None


def _safe_call(function, *arguments):

    try:
        return function(*arguments)
    except Exception:
        logger.exception("Optional external step failed")
        return None


def _build_forecast_fallback(latitude, climatology):
    """Fallback 16-day forecast + FAO ET0 if Open-Meteo is unreachable."""
    today = date.today()
    days = []
    for offset in range(FORECAST_DAYS):
        d = today + timedelta(days=offset)
        clim_m = (climatology or {}).get(d.month) or _EGYPT_MONTHLY_NORMALS[d.month]
        t_mean = clim_m.get("temp") or 24.0
        t_max = round(t_mean + 5.5, 1)
        t_min = round(t_mean - 5.5, 1)
        solar = clim_m.get("solar") or 19.0
        et0 = round(max(1.8, 0.0023 * (t_mean + 17.8) * math.sqrt(max(1.0, t_max - t_min)) * (solar * 0.408)), 2)
        days.append({
            "date": d.isoformat(),
            "t_max": t_max,
            "t_min": t_min,
            "rain_mm": 0.0,
            "wind_max_kmh": round((clim_m.get("ws") or 3.6) * 3.6, 1),
            "et0_mm": et0,
        })
    return {"days": days, "elevation": None, "is_fallback": True}


def get_short_forecast(latitude, longitude):

    cache_key = ("short_forecast_v2", *_coord_cache_key(latitude, longitude))
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    data = _open_meteo_get(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,wind_speed_10m_max,et0_fao_evapotranspiration"
            ),
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
        }
    )

    try:
        daily = data["daily"]
        times = daily["time"]
        t_max_list = daily["temperature_2m_max"]
        t_min_list = daily["temperature_2m_min"]
        rain_list = daily.get("precipitation_sum") or []
        wind_list = daily.get("wind_speed_10m_max") or []
        et0_list = daily.get("et0_fao_evapotranspiration") or []

        days = []

        for index, day in enumerate(times):
            if index >= len(t_max_list) or index >= len(t_min_list):
                break

            t_max = _climatology_value(t_max_list[index])
            t_min = _climatology_value(t_min_list[index])
            if t_max is None or t_min is None:
                continue

            rain = _climatology_value(rain_list[index] if index < len(rain_list) else None)
            wind = _climatology_value(wind_list[index] if index < len(wind_list) else None)
            et0 = _climatology_value(et0_list[index] if index < len(et0_list) else None)

            if et0 is None:
                t_mean = (t_max + t_min) / 2.0
                et0 = round(max(1.5, 0.16 * max(5.0, t_mean)), 2)

            days.append({
                "date": str(day),
                "t_max": t_max,
                "t_min": t_min,
                "rain_mm": rain if rain is not None else 0.0,
                "wind_max_kmh": wind,
                "et0_mm": round(et0, 2),
            })

        if days:
            result = {
                "days": days,
                "elevation": _climatology_value(data.get("elevation")),
                "is_fallback": False,
            }
            _DATA_CACHE.set(cache_key, result, FORECAST_CACHE_TTL)
            return result

        return None

    except (KeyError, TypeError, IndexError, AttributeError):
        return None


def _parse_outlook(data):

    if not data:
        return None

    try:
        daily = data["daily"]
        times = daily["time"]

        def mean_series(variable):
            series = [
                values for key, values in daily.items()
                if key == variable or key.startswith(variable + "_member")
            ]
            if not series:
                return None

            output = []
            for index in range(len(times)):
                values = [
                    _climatology_value(item[index])
                    for item in series if index < len(item)
                ]
                values = [v for v in values if v is not None]
                output.append(sum(values) / len(values) if values else None)
            return output

        t_max = mean_series("temperature_2m_max")
        t_min = mean_series("temperature_2m_min")
        rain = mean_series("precipitation_sum")

        if t_max is None or t_min is None:
            return None

        result = {}
        for index, day in enumerate(times):
            if t_max[index] is None or t_min[index] is None:
                continue
            result[str(day)] = {
                "temp": (t_max[index] + t_min[index]) / 2,
                "rain": (
                    rain[index]
                    if rain and rain[index] is not None else 0.0
                ),
            }

        return result or None

    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def get_outlook(latitude, longitude):

    cache_key = ("outlook", *_coord_cache_key(latitude, longitude))
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    base = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }

    data = _open_meteo_get(OPEN_METEO_SEASONAL_URL, {**base, "forecast_days": 46})
    parsed = _parse_outlook(data)
    if parsed:
        _DATA_CACHE.set(cache_key, parsed, OUTLOOK_CACHE_TTL)
        return parsed

    return None


# ============================================================
# 13F. LIVE USD/EGP EXCHANGE RATE & COMMODITY/FERTILIZER INDEX
# ============================================================

def get_live_market_indicators():

    cache_key = ("market_indicators",)
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    session = _get_http_session()
    usd_egp = FALLBACK_USD_EGP
    source = "Baseline Market Fallback"
    updated_utc = date.today().isoformat()
    is_fallback = True

    try:
        resp = session.get(EXCHANGE_RATE_URL, timeout=EXTERNAL_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates") or {}
        egp_rate = _safe_float(rates.get("EGP"))
        if egp_rate is not None and egp_rate > 10:
            usd_egp = round(egp_rate, 2)
            source = "open.er-api.com (Live USD/EGP)"
            updated_utc = data.get("time_last_update_utc") or updated_utc
            is_fallback = False
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as error:
        logger.warning("Exchange rate API failed (%s); using fallback rate %.2f.", error, FALLBACK_USD_EGP)

    fx_ratio = usd_egp / BASELINE_USD_EGP
    fertilizer_global_index = 1.04
    cost_index_factor = round((0.40 + 0.60 * fx_ratio) * (0.75 + 0.25 * fertilizer_global_index), 4)
    price_index_factor = round(0.55 + 0.45 * fx_ratio, 4)

    result = {
        "usd_egp_rate": usd_egp,
        "baseline_usd_egp": BASELINE_USD_EGP,
        "fx_ratio": round(fx_ratio, 4),
        "fertilizer_global_index": fertilizer_global_index,
        "cost_index_factor": cost_index_factor,
        "price_index_factor": price_index_factor,
        "source": source,
        "updated_utc": updated_utc,
        "is_fallback": is_fallback,
    }

    if not is_fallback:
        _DATA_CACHE.set(cache_key, result, MARKET_CACHE_TTL)

    return result


# ============================================================
# 14. CLEAN NASA VALUES
# ============================================================

def clean_values(parameter_data):

    if not isinstance(parameter_data, dict):
        return []

    valid_values = []

    for value in parameter_data.values():

        if value is None:
            continue

        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            continue

        if not math.isfinite(numeric_value) or numeric_value == -999.0:
            continue

        valid_values.append(numeric_value)

    return valid_values


# ============================================================
# 15. CALCULATE WEATHER & SOIL MOISTURE SUMMARY
# ============================================================

def calculate_weather_summary(weather_data):

    temperature = clean_values(weather_data.get("T2M", {}))
    rainfall = clean_values(weather_data.get("PRECTOTCORR", {}))
    humidity = clean_values(weather_data.get("RH2M", {}))
    wind_speed = clean_values(weather_data.get("WS10M", {}))
    solar_rad = clean_values(weather_data.get("ALLSKY_SFC_SW_DWN", {}))
    gwet_top = clean_values(weather_data.get("GWETTOP", {}))
    gwet_root = clean_values(weather_data.get("GWETROOT", {}))
    evptrns = clean_values(weather_data.get("EVPTRNS", {}))

    if not temperature:
        temperature = [22.0]
    if not rainfall:
        rainfall = [0.0]

    total_days = len(weather_data.get("T2M", {})) or WEATHER_PERIOD_DAYS

    avg_temp = round(mean(temperature), 2)
    avg_solar = round(mean(solar_rad), 2) if solar_rad else None
    avg_gwet_top = round(mean(gwet_top), 3) if gwet_top else None
    avg_gwet_root = round(mean(gwet_root), 3) if gwet_root else None
    avg_evptrns_w_m2 = round(mean(evptrns), 2) if evptrns else None

    if avg_evptrns_w_m2 is not None and avg_evptrns_w_m2 > 0:
        est_daily_et_mm = round(avg_evptrns_w_m2 / 28.35, 2)
    elif avg_solar is not None:
        est_daily_et_mm = round(max(1.5, 0.0135 * (avg_temp + 17.78) * (avg_solar * 0.408)), 2)
    else:
        est_daily_et_mm = round(max(1.8, 0.17 * max(5.0, avg_temp)), 2)

    data_quality = {
        "temperature": {
            "valid_days": len(temperature),
            "missing_days": max(0, total_days - len(temperature)),
            "total_days": total_days,
        },
        "rainfall": {
            "valid_days": len(rainfall),
            "missing_days": max(0, total_days - len(rainfall)),
            "total_days": total_days,
        },
        "humidity": {
            "valid_days": len(humidity),
            "missing_days": max(0, total_days - len(humidity)),
            "total_days": total_days,
        },
        "wind_speed": {
            "valid_days": len(wind_speed),
            "missing_days": max(0, total_days - len(wind_speed)),
            "total_days": total_days,
        },
        "solar_radiation": {
            "valid_days": len(solar_rad),
            "missing_days": max(0, total_days - len(solar_rad)),
            "total_days": total_days,
        },
        "soil_wetness_smap": {
            "valid_days": len(gwet_root),
            "missing_days": max(0, total_days - len(gwet_root)),
            "total_days": total_days,
        },
        "nasa_live_status": "fallback" if weather_data.get("_is_fallback") else "live",
    }

    return {
        "average_temperature_c": avg_temp,
        "average_daily_rainfall_mm": round(mean(rainfall), 2),
        "total_rainfall_mm": round(sum(rainfall), 2),
        "average_relative_humidity_percent": (
            round(mean(humidity), 2) if humidity else None
        ),
        "average_wind_speed_m_s": (
            round(mean(wind_speed), 2) if wind_speed else None
        ),
        "average_solar_radiation_mj_m2_day": avg_solar,
        "average_surface_soil_wetness": avg_gwet_top,
        "average_root_zone_soil_wetness": avg_gwet_root,
        "average_evapotranspiration_w_m2": avg_evptrns_w_m2,
        "estimated_daily_et_mm": est_daily_et_mm,
        "days_with_valid_temperature_data": len(temperature),
        "days_with_valid_rainfall_data": len(rainfall),
        "data_quality": data_quality,
    }


# ============================================================
# 16. CROP REQUIREMENTS, ECONOMICS & CALENDAR TABLES
# ============================================================

def _crop(en, temp_opt, temp_abs, soil, months, water, waterlog, salt,
          irrigation, humid_limit=None, fresh=False, wind_sensitive=False,
          kc=0.90, ph_range=(6.0, 8.2), risks=(), tips=()):

    soil_ratings = dict(zip(("رملية", "طينية", "طميية"), soil))

    return {
        "en": en,
        "temp_opt": temp_opt,
        "temp_abs": temp_abs,
        "soil": soil_ratings,
        "months": months,
        "water": water,
        "waterlog": waterlog,
        "salt": salt,
        "irrigation": list(irrigation),
        "humid_limit": humid_limit,
        "fresh": fresh,
        "wind_sensitive": wind_sensitive,
        "kc": kc,
        "ph_range": ph_range,
        "risks": list(risks),
        "tips": list(tips),
    }


CROP_PROFILES = {

    "القمح": _crop(
        "Wheat", (12, 22), (3, 32), "tpp", [11, 12],
        "medium", "low", "medium", ["الرش", "الغمر"], humid_limit=75, kc=0.85, ph_range=(6.0, 8.3),
        risks=[
            ("الصدأ الأصفر والبني يزداد مع الرطوبة العالية والجو البارد",
             "Yellow and brown rust increase with high humidity and cool weather"),
            ("رقاد النباتات مع الإفراط في التسميد الآزوتي والري الغزير",
             "Lodging with excess nitrogen and heavy irrigation"),
        ],
        tips=[
            ("ازرع في الميعاد المناسب (نوفمبر) لأن التأخير يقلل المحصول",
             "Sow on time (November); late sowing reduces yield"),
            ("نظّم الريات حسب مراحل النمو وتجنب الري الغزير قبل الحصاد",
             "Schedule irrigations by growth stage and avoid heavy irrigation near harvest"),
        ],
    ),

    "الأرز": _crop(
        "Rice", (22, 32), (15, 38), "upt", [4, 5, 6],
        "very_high", "tolerant", "low", ["الغمر"], kc=1.15, ph_range=(5.5, 8.0),
        risks=[
            ("استهلاك مياه مرتفع جدًا، وزراعة الأرز منظمة قانونيًا في مناطق محددة",
             "Very high water use; rice growing is legally restricted to specific zones"),
            ("ملوحة التربة أو مياه الري تضر الشتلات الصغيرة",
             "Salinity of soil or irrigation water damages young seedlings"),
        ],
        tips=[
            ("تأكد أن أرضك داخل المناطق المسموح فيها بزراعة الأرز قبل الزراعة",
             "Confirm your land is inside the permitted rice zones before planting"),
            ("يمكن توفير المياه بالري المتقطع بعد استشارة الإرشاد الزراعي",
             "Water can be saved with alternate wetting and drying after consulting extension services"),
        ],
    ),

    "الذرة": _crop(
        "Maize", (18, 32), (10, 38), "ttp", [4, 5, 6, 7],
        "high", "low", "low", ["الرش", "التنقيط", "الغمر"], kc=0.95, ph_range=(5.8, 8.0),
        risks=[
            ("العطش وقت التزهير والتلقيح يخفض المحصول بشدة",
             "Water stress at tasselling and silking sharply reduces yield"),
            ("الحشرات الثاقبة وديدان الحشد من الآفات الشائعة وتحتاج متابعة مبكرة",
             "Stem borers and armyworms are common pests and need early scouting"),
        ],
        tips=[
            ("نظّم الري قبل التزهير وأثناءه ولا تترك الحقل يعطش",
             "Irrigate regularly before and during flowering"),
            ("الزراعة في الميعاد تقلل الإصابة بالآفات وتجنّب موجات الحر",
             "Timely planting reduces pest pressure and heat-wave damage"),
        ],
    ),

    "القطن": _crop(
        "Cotton", (21, 32), (15, 38), "tpp", [3, 4],
        "high", "low", "high", ["الغمر", "التنقيط"], kc=0.92, ph_range=(6.0, 8.4),
        risks=[
            ("دودة ورق القطن والديدان القرنفلية تحتاج متابعة مستمرة",
             "Cotton leafworm and bollworms need continuous scouting"),
            ("الإفراط في الري والتسميد الآزوتي يزيد النمو الخضري ويؤخر التفتح",
             "Excess water and nitrogen promote leaf growth and delay boll opening"),
        ],
        tips=[
            ("التزم بمواعيد الزراعة التي يحددها الإرشاد الزراعي لأن القطن حساس لميعاد الزراعة",
             "Follow extension-service planting dates; cotton is sensitive to sowing date"),
            ("قلّل الريات في نهاية الموسم لتشجيع تفتح اللوز",
             "Reduce irrigation late in the season to encourage boll opening"),
        ],
    ),

    "قصب السكر": _crop(
        "Sugarcane", (24, 32), (15, 38), "tpp", [2, 3, 9, 10],
        "very_high", "medium", "medium", ["الغمر", "التنقيط"], kc=1.10, ph_range=(6.0, 8.2),
        risks=[
            ("استهلاك مياه مرتفع جدًا وهو عبء كبير عند نقص المياه",
             "Very high water use, a heavy burden when water is short"),
            ("ثاقبات الساق من الآفات الشائعة",
             "Stalk borers are common pests"),
        ],
        tips=[
            ("اختر الأصناف الموصى بها من مركز البحوث الزراعية",
             "Use varieties recommended by the Agricultural Research Center"),
            ("حافظ على الصرف الجيد لأن ارتفاع الماء الأرضي والملوحة يخفضان المحصول",
             "Keep good drainage; a high water table and salinity reduce yield"),
        ],
    ),

    "بنجر السكر": _crop(
        "Sugar beet", (15, 25), (5, 32), "ttp", [9, 10, 11],
        "medium", "low", "high", ["الرش", "التنقيط", "الغمر"], humid_limit=75, kc=0.88, ph_range=(6.2, 8.5),
        risks=[
            ("تبقع الأوراق (سركوسبورا) يزداد مع الرطوبة العالية",
             "Cercospora leaf spot increases with high humidity"),
            ("الري الزائد قرب النضج يقلل نسبة السكر",
             "Excess irrigation near maturity lowers sugar content"),
        ],
        tips=[
            ("يتحمل الملوحة نسبيًا لذلك يناسب الأراضي الأقل جودة",
             "Relatively salt-tolerant, so it suits marginal soils"),
            ("ازرع في ميعاد الخريف الموصى به لتجنب الحرارة المرتفعة",
             "Plant in the recommended autumn window to avoid high temperatures"),
        ],
    ),

    "الطماطم": _crop(
        "Tomato", (18, 27), (10, 35), "ttp", [2, 3, 7, 8, 10],
        "high", "very_low", "medium", ["التنقيط"], humid_limit=70, fresh=True, kc=0.95, ph_range=(5.8, 7.8),
        risks=[
            ("الذبول وعفن الجذور بسبب تجمع المياه حول الجذور",
             "Wilt and root rot from standing water around the roots"),
            ("اللفحة المتأخرة والأمراض الفطرية تزيد مع الرطوبة العالية",
             "Late blight and fungal diseases increase with high humidity"),
            ("تساقط الأزهار وضعف العقد عند ارتفاع الحرارة فوق 35 درجة",
             "Flower drop and poor fruit set above 35 degrees C"),
        ],
        tips=[
            ("استخدم الري بالتنقيط وتجنب بلل الأوراق",
             "Use drip irrigation and avoid wetting the leaves"),
            ("ازرع على خطوط مرتفعة وفي أرض جيدة الصرف",
             "Plant on raised beds in well-drained soil"),
        ],
    ),

    "البطاطس": _crop(
        "Potato", (15, 22), (7, 28), "ptp", [1, 2, 8, 9],
        "high", "very_low", "low", ["التنقيط", "الرش"], humid_limit=75, fresh=True, kc=0.92, ph_range=(5.5, 7.6),
        risks=[
            ("اللفحة المتأخرة مع الرطوبة العالية والجو البارد",
             "Late blight with high humidity and cool weather"),
            ("تعفن الدرنات عند تجمع المياه أو الري الزائد",
             "Tuber rot with standing water or over-irrigation"),
        ],
        tips=[
            ("استخدم تقاوي سليمة ومعتمدة",
             "Use certified, healthy seed tubers"),
            ("حافظ على رطوبة منتظمة للتربة لتجنب تشقق الدرنات",
             "Keep soil moisture even to avoid tuber cracking"),
        ],
    ),

    "البصل": _crop(
        "Onion", (13, 24), (7, 30), "ttp", [9, 10, 11, 12],
        "medium", "low", "low", ["التنقيط", "الرش", "الغمر"], humid_limit=75, fresh=True, kc=0.82, ph_range=(6.0, 7.8),
        risks=[
            ("البياض الزغبي والعفن مع الرطوبة العالية",
             "Downy mildew and rots with high humidity"),
            ("الملوحة تقلل الإنتاج بشدة لأن البصل حساس لها",
             "Salinity strongly reduces yield because onion is sensitive"),
        ],
        tips=[
            ("قلّل الري قبل الحصاد بفترة لتحسين قابلية التخزين",
             "Reduce irrigation before harvest to improve storage"),
            ("ازرع في الميعاد المناسب لتجنب الإزهار المبكر",
             "Plant on time to avoid premature bolting"),
        ],
    ),

    "الثوم": _crop(
        "Garlic", (12, 24), (5, 30), "ttp", [9, 10],
        "medium", "low", "low", ["التنقيط", "الرش", "الغمر"], humid_limit=75, fresh=True, kc=0.80, ph_range=(6.0, 7.8),
        risks=[
            ("العفن الأبيض والأمراض الفطرية مع الرطوبة وتجمع الماء",
             "White rot and fungal diseases with moisture and standing water"),
            ("حساس للملوحة",
             "Sensitive to salinity"),
        ],
        tips=[
            ("استخدم فصوصًا سليمة للتقاوي",
             "Use healthy cloves for planting"),
            ("أوقف الري قبل الحصاد بفترة كافية",
             "Stop irrigation sufficiently long before harvest"),
        ],
    ),

    "الفول البلدي": _crop(
        "Faba bean", (10, 22), (3, 30), "upp", [10, 11],
        "medium", "low", "low", ["الغمر", "الرش"], humid_limit=75, kc=0.80, ph_range=(6.2, 8.2),
        risks=[
            ("طفيل الهالوك من أخطر مشكلات الفول في مصر",
             "Broomrape (Orobanche) is a major faba bean problem in Egypt"),
            ("التبقع الشكولاتي مع الرطوبة العالية",
             "Chocolate spot with high humidity"),
        ],
        tips=[
            ("اختر الأصناف المقاومة للهالوك الموصى بها",
             "Choose recommended broomrape-resistant varieties"),
            ("تجنب الري الزائد خصوصًا في الأراضي الثقيلة",
             "Avoid over-irrigation, especially in heavy soils"),
        ],
    ),

    "الفاصوليا": _crop(
        "Common bean", (16, 27), (10, 32), "ttp", [2, 3, 8, 9],
        "medium", "low", "low", ["التنقيط", "الرش"], humid_limit=75, fresh=True, kc=0.85, ph_range=(6.0, 7.6),
        risks=[
            ("حساسة جدًا للملوحة والبرودة والحرارة العالية",
             "Very sensitive to salinity, cold and high heat"),
            ("الأمراض الفطرية مع الرطوبة العالية",
             "Fungal diseases with high humidity"),
        ],
        tips=[
            ("ازرع في ميعاد معتدل الحرارة",
             "Plant in a mild-temperature window"),
            ("اروِ بانتظام بالتنقيط أو بالرش الخفيف",
             "Irrigate regularly with drip or light sprinklers"),
        ],
    ),

    "العدس": _crop(
        "Lentil", (10, 24), (3, 30), "ptp", [11, 12],
        "low", "low", "low", ["الرش", "الغمر"], humid_limit=75, kc=0.72, ph_range=(6.0, 8.2),
        risks=[
            ("الذبول والأمراض الفطرية مع الرطوبة وتجمع الماء",
             "Wilt and fungal diseases with moisture and standing water"),
            ("منافسة الحشائش في بداية النمو",
             "Weed competition early in growth"),
        ],
        tips=[
            ("محصول يتحمل قلة المياه فلا تفرط في الري",
             "Drought-tolerant; avoid over-irrigation"),
            ("ازرع في أرض جيدة الصرف",
             "Plant in well-drained soil"),
        ],
    ),

    "الحمص": _crop(
        "Chickpea", (10, 26), (5, 32), "ptp", [11, 12],
        "low", "very_low", "low", ["الرش", "التنقيط"], humid_limit=75, kc=0.72, ph_range=(6.0, 8.2),
        risks=[
            ("لفحة الأسكوكيتا مع الرطوبة العالية",
             "Ascochyta blight with high humidity"),
            ("لا يتحمل تجمع الماء إطلاقًا",
             "Does not tolerate waterlogging at all"),
        ],
        tips=[
            ("ري محدود وعلى فترات متباعدة",
             "Limited irrigation at wide intervals"),
            ("استخدم تقاوي سليمة ومعاملة",
             "Use healthy, treated seed"),
        ],
    ),

    "الفول السوداني": _crop(
        "Peanut", (22, 30), (15, 35), "pup", [4, 5],
        "medium", "low", "low", ["الرش", "التنقيط"], kc=0.88, ph_range=(5.8, 7.8),
        risks=[
            ("التربة الثقيلة تعيق تكوّن القرون",
             "Heavy soil hampers pod development"),
            ("تلوث القرون بالأفلاتوكسين عند الجفاف أو سوء التخزين",
             "Aflatoxin contamination under drought stress or poor storage"),
        ],
        tips=[
            ("انتظم في الري أثناء تكوّن القرون",
             "Irrigate regularly during pod formation"),
            ("جفّف المحصول جيدًا بعد الحصاد وخزّنه جافًا",
             "Dry the harvest thoroughly and store it dry"),
        ],
    ),

    "عباد الشمس": _crop(
        "Sunflower", (18, 28), (8, 35), "ttp", [3, 4, 7, 8],
        "medium", "low", "medium", ["الرش", "الغمر", "التنقيط"], kc=0.86, ph_range=(6.0, 8.2),
        risks=[
            ("العطش وقت التزهير يخفض المحصول",
             "Water stress at flowering lowers yield"),
            ("الطيور والأمراض الفطرية في الجو الرطب",
             "Birds and fungal diseases in humid weather"),
        ],
        tips=[
            ("التزم بكثافة الزراعة الموصى بها",
             "Use the recommended plant density"),
            ("اروِ في مرحلتي التزهير وامتلاء البذور",
             "Irrigate at flowering and seed filling"),
        ],
    ),

    "السمسم": _crop(
        "Sesame", (25, 32), (18, 38), "ttp", [5, 6],
        "low", "very_low", "low", ["الرش", "الغمر"], kc=0.75, ph_range=(5.8, 8.0),
        risks=[
            ("لا يتحمل تجمع الماء أبدًا",
             "Cannot tolerate waterlogging"),
            ("الرطوبة العالية أو الأمطار وقت النضج تقلل الجودة",
             "High humidity or rain at maturity reduces quality"),
        ],
        tips=[
            ("يتحمل العطش نسبيًا فلا تسرف في الري",
             "Fairly drought-tolerant; do not over-irrigate"),
            ("ازرع في أرض جيدة الصرف",
             "Plant in well-drained soil"),
        ],
    ),

    "البرسيم": _crop(
        "Berseem clover", (10, 25), (3, 32), "upp", [9, 10, 11],
        "high", "medium", "medium", ["الغمر"], kc=0.95, ph_range=(6.2, 8.2),
        risks=[
            ("ملوحة التربة تخفض الإنتاج",
             "Soil salinity reduces yield"),
            ("الحامول (الكشوت) حشيشة طفيلية تضر المحصول",
             "Dodder is a parasitic weed that damages the crop"),
        ],
        tips=[
            ("استخدم بذورًا نظيفة خالية من بذور الحامول",
             "Use clean seed free of dodder seed"),
            ("اروِ بعد كل حشّة",
             "Irrigate after each cut"),
        ],
    ),

    "العنب": _crop(
        "Grapes", (20, 30), (10, 38), "ptp", [1, 2, 3],
        "medium", "low", "medium", ["التنقيط"], humid_limit=70, fresh=True, kc=0.75, ph_range=(6.0, 8.0),
        risks=[
            ("البياض الدقيقي والزغبي مع الرطوبة العالية",
             "Powdery and downy mildew with high humidity"),
            ("تشقق الحبات بسبب الري غير المنتظم",
             "Berry cracking from irregular irrigation"),
        ],
        tips=[
            ("نظّم التقليم والتربية ليدخل الهواء بين الأفرع",
             "Prune and train vines for good airflow"),
            ("استخدم الري بالتنقيط",
             "Use drip irrigation"),
        ],
    ),

    "المانجو": _crop(
        "Mango", (24, 30), (15, 40), "pup", [3, 4, 5],
        "medium", "low", "low", ["التنقيط"], humid_limit=70, fresh=True, kc=0.80, ph_range=(5.8, 7.8),
        risks=[
            ("البرد الشديد والصقيع يضران الأشجار الصغيرة والإزهار",
             "Strong cold and frost damage young trees and flowering"),
            ("البياض الدقيقي والأنثراكنوز مع الرطوبة وقت الإزهار",
             "Powdery mildew and anthracnose in humid weather at flowering"),
        ],
        tips=[
            ("وفّر مصدات رياح وحماية للشتلات الصغيرة في الشتاء",
             "Provide windbreaks and winter protection for young trees"),
            ("اتبع توصية الإرشاد الزراعي في تنظيم الري قبل الإزهار",
             "Follow extension-service advice on irrigation before flowering"),
        ],
    ),

    "الموالح": _crop(
        "Citrus", (15, 30), (7, 38), "ttp", [2, 3, 4, 9, 10],
        "high", "low", "low", ["التنقيط", "الغمر"], fresh=True, kc=0.78, ph_range=(6.0, 7.8),
        risks=[
            ("حساسة للملوحة وتجمع الماء",
             "Sensitive to salinity and waterlogging"),
            ("الصقيع قد يضر الثمار والأشجار الصغيرة",
             "Frost can damage fruit and young trees"),
        ],
        tips=[
            ("سمّد بانتظام وتجنب الإفراط في الري",
             "Fertilize regularly and avoid over-irrigation"),
            ("استخدم أصولًا مناسبة لنوع تربتك",
             "Use rootstocks suited to your soil"),
        ],
    ),

    "الموز": _crop(
        "Banana", (22, 30), (12, 35), "ttp", [3, 4, 5],
        "very_high", "low", "low", ["التنقيط", "الرش"], fresh=True, wind_sensitive=True, kc=1.05, ph_range=(6.0, 7.8),
        risks=[
            ("البرد الشتوي والرياح الشديدة تضر النباتات",
             "Winter cold and strong winds damage the plants"),
            ("استهلاك مياه مرتفع جدًا",
             "Very high water demand"),
        ],
        tips=[
            ("ازرع مصدات رياح حول المزرعة",
             "Plant windbreaks around the farm"),
            ("حافظ على رطوبة منتظمة وصرف جيد",
             "Keep soil moisture even with good drainage"),
        ],
    ),

    "الزيتون": _crop(
        "Olive", (15, 30), (5, 40), "ptp", [2, 3, 10, 11],
        "low", "low", "high", ["التنقيط"], humid_limit=75, kc=0.62, ph_range=(6.2, 8.5),
        risks=[
            ("عين الطاووس مع الرطوبة العالية",
             "Peacock spot with high humidity"),
            ("الصقيع وقت الإزهار",
             "Frost at flowering"),
        ],
        tips=[
            ("يتحمل العطش والملوحة نسبيًا لكنه يحتاج صرفًا جيدًا",
             "Fairly drought- and salt-tolerant but needs good drainage"),
            ("التقليم السنوي يحسّن التهوية والإنتاج",
             "Annual pruning improves airflow and yield"),
        ],
    ),

    "الفراولة": _crop(
        "Strawberry", (15, 25), (7, 30), "pup", [9, 10],
        "high", "very_low", "low", ["التنقيط"], humid_limit=70, fresh=True, kc=0.82, ph_range=(5.5, 7.4),
        risks=[
            ("العفن الرمادي مع الرطوبة العالية",
             "Gray mold with high humidity"),
            ("شديدة الحساسية للملوحة وتجمع الماء",
             "Very sensitive to salinity and waterlogging"),
        ],
        tips=[
            ("استخدم الري بالتنقيط مع تغطية بلاستيكية",
             "Use drip irrigation with plastic mulch"),
            ("استخدم شتلات سليمة خالية من الأمراض",
             "Use healthy, disease-free transplants"),
        ],
    ),

}


CROP_ECONOMICS = {
    "القمح": {"cost": 18000, "yield_ton": 2.7, "price_per_ton": 13500, "water_flood_m3": 2200, "duration_days": 155, "category": "field"},
    "الأرز": {"cost": 24000, "yield_ton": 4.0, "price_per_ton": 14500, "water_flood_m3": 6200, "duration_days": 135, "category": "rice"},
    "الذرة": {"cost": 20000, "yield_ton": 3.5, "price_per_ton": 12500, "water_flood_m3": 3100, "duration_days": 115, "category": "field"},
    "القطن": {"cost": 26000, "yield_ton": 1.4, "price_per_ton": 48000, "water_flood_m3": 3600, "duration_days": 185, "category": "field"},
    "قصب السكر": {"cost": 45000, "yield_ton": 48.0, "price_per_ton": 2200, "water_flood_m3": 9500, "duration_days": 365, "category": "sugarcane"},
    "بنجر السكر": {"cost": 24000, "yield_ton": 22.0, "price_per_ton": 2400, "water_flood_m3": 2800, "duration_days": 190, "category": "root"},
    "الطماطم": {"cost": 55000, "yield_ton": 24.0, "price_per_ton": 5500, "water_flood_m3": 3200, "duration_days": 120, "category": "vegetable"},
    "البطاطس": {"cost": 65000, "yield_ton": 14.0, "price_per_ton": 9000, "water_flood_m3": 2700, "duration_days": 110, "category": "root"},
    "البصل": {"cost": 38000, "yield_ton": 15.0, "price_per_ton": 6500, "water_flood_m3": 2500, "duration_days": 150, "category": "root"},
    "الثوم": {"cost": 42000, "yield_ton": 10.0, "price_per_ton": 11000, "water_flood_m3": 2400, "duration_days": 170, "category": "root"},
    "الفول البلدي": {"cost": 17000, "yield_ton": 1.6, "price_per_ton": 26000, "water_flood_m3": 1800, "duration_days": 150, "category": "legume"},
    "الفاصوليا": {"cost": 28000, "yield_ton": 1.35, "price_per_ton": 45000, "water_flood_m3": 2300, "duration_days": 100, "category": "legume"},
    "العدس": {"cost": 16000, "yield_ton": 0.9, "price_per_ton": 38000, "water_flood_m3": 1500, "duration_days": 135, "category": "legume"},
    "الحمص": {"cost": 16500, "yield_ton": 1.0, "price_per_ton": 36000, "water_flood_m3": 1500, "duration_days": 140, "category": "legume"},
    "الفول السوداني": {"cost": 26000, "yield_ton": 1.8, "price_per_ton": 35000, "water_flood_m3": 2900, "duration_days": 140, "category": "legume"},
    "عباد الشمس": {"cost": 15000, "yield_ton": 1.2, "price_per_ton": 24000, "water_flood_m3": 2200, "duration_days": 100, "category": "field"},
    "السمسم": {"cost": 16000, "yield_ton": 0.7, "price_per_ton": 52000, "water_flood_m3": 1900, "duration_days": 115, "category": "field"},
    "البرسيم": {"cost": 14000, "yield_ton": 28.0, "price_per_ton": 1400, "water_flood_m3": 3200, "duration_days": 180, "category": "forage"},
    "العنب": {"cost": 65000, "yield_ton": 10.0, "price_per_ton": 15000, "water_flood_m3": 3600, "duration_days": 365, "category": "orchard"},
    "المانجو": {"cost": 55000, "yield_ton": 7.5, "price_per_ton": 20000, "water_flood_m3": 4200, "duration_days": 365, "category": "orchard"},
    "الموالح": {"cost": 50000, "yield_ton": 13.0, "price_per_ton": 9500, "water_flood_m3": 4000, "duration_days": 365, "category": "orchard"},
    "الموز": {"cost": 85000, "yield_ton": 20.0, "price_per_ton": 11500, "water_flood_m3": 7200, "duration_days": 365, "category": "orchard"},
    "الزيتون": {"cost": 32000, "yield_ton": 4.0, "price_per_ton": 22000, "water_flood_m3": 2200, "duration_days": 365, "category": "orchard"},
    "الفراولة": {"cost": 130000, "yield_ton": 18.0, "price_per_ton": 16000, "water_flood_m3": 3800, "duration_days": 210, "category": "vegetable"},
}


SOIL_TIPS = {

    "رملية": (
        "التربة الرملية تفقد الماء والعناصر بسرعة: اروِ على دفعات صغيرة "
        "متكررة، وأضف مادة عضوية، وسمّد على دفعات.",
        "Sandy soil loses water and nutrients quickly: irrigate in small "
        "frequent amounts, add organic matter and split fertilizer doses."
    ),

    "طينية": (
        "التربة الطينية بطيئة الصرف: تجنب الري الغزير المتكرر، وازرع على "
        "خطوط مرتفعة، وحافظ على نظافة المصارف.",
        "Clay soil drains slowly: avoid frequent heavy irrigation, plant on "
        "raised beds and keep drains clean."
    ),

    "طميية": (
        "التربة الطميية جيدة عمومًا: حافظ على مادتها العضوية وصرفها.",
        "Loam soil is generally good: maintain its organic matter and drainage."
    ),

}


PROBLEM_ADVICE = {

    "نقص المياه": [
        ("حوّل إلى الري بالتنقيط إن أمكن، واروِ في الصباح الباكر أو المساء.",
         "Switch to drip irrigation if possible and irrigate early morning or evening."),
        ("استخدم التغطية (المالش) لتقليل البخر من التربة.",
         "Use mulch to reduce evaporation from the soil."),
    ],

    "ملوحة التربة": [
        ("اعمل تحليل ملوحة (EC) للتربة ولمياه الري عند معمل معتمد.",
         "Test the salinity (EC) of the soil and irrigation water in an accredited lab."),
        ("حسّن الصرف واغسل الأملاح بريّة غسيل حسب توصية المهندس الزراعي.",
         "Improve drainage and leach salts as advised by an agricultural engineer."),
        ("اختر محاصيل تتحمل الملوحة مثل بنجر السكر والقطن والزيتون.",
         "Choose salt-tolerant crops such as sugar beet, cotton and olive."),
    ],

    "اصفرار النباتات": [
        ("الاصفرار قد يكون من نقص النيتروجين أو الحديد، أو من تشبع التربة بالماء، أو من الملوحة.",
         "Yellowing can come from nitrogen or iron deficiency, waterlogging or salinity."),
        ("افحص الجذور وحلّل التربة قبل إضافة أي أسمدة.",
         "Check the roots and test the soil before adding any fertilizer."),
    ],

    "الآفات والحشرات": [
        ("افحص الحقل بانتظام وسجّل أماكن الإصابة.",
         "Scout the field regularly and record infested areas."),
        ("اعرض عينة على الإرشاد الزراعي لتحديد الآفة قبل استخدام أي مبيد.",
         "Show a sample to the agricultural extension service to identify the pest before using any pesticide."),
    ],

    "الأمراض النباتية": [
        ("قلّل بلل الأوراق وحسّن التهوية بين النباتات.",
         "Reduce leaf wetness and improve airflow between plants."),
        ("اعرض عينة على الإرشاد الزراعي لتحديد المرض واتباع العلاج الموصى به.",
         "Show a sample to the extension service to identify the disease and follow the recommended treatment."),
    ],

    "ضعف نمو المحصول": [
        ("راجع الري والتسميد وحموضة التربة وملوحتها.",
         "Review irrigation, fertilization, soil pH and salinity."),
        ("حلّل التربة لتحديد سبب الضعف قبل تغيير البرنامج.",
         "Test the soil to find the cause before changing the program."),
    ],

    "مشكلة أخرى": [
        ("اكتب تفاصيل المشكلة في الملاحظات واعرضها على الإرشاد الزراعي.",
         "Describe the problem in the notes and show it to the extension service."),
    ],

    "لا توجد مشكلة": [],

}


IRRIGATION_REASONS = {

    "التنقيط": (
        "يوفّر المياه ويوصلها للجذور مباشرة ويقلل تجمع المياه والأمراض.",
        "It saves water, delivers it to the roots and reduces standing water and disease."
    ),

    "الرش": (
        "مناسب للمحاصيل الحقلية ويوزع الماء بانتظام، مع الحذر من ترطيب الأوراق في الجو الرطب.",
        "It suits field crops and spreads water evenly; avoid wetting leaves in humid weather."
    ),

    "الغمر": (
        "طريقة تقليدية مناسبة للمحاصيل التي تتحمل الماء، لكنها تستهلك مياهًا أكثر.",
        "A traditional method that suits water-tolerant crops, but it uses more water."
    ),

}


# ============================================================
# 17. HELPERS FOR TEXT, MODELS AND MONTHS
# ============================================================

LABELS_EN = {

    "رملية": "sandy",
    "طينية": "clay",
    "طميية": "loam",

    "الغمر": "flood irrigation",
    "التنقيط": "drip irrigation",
    "الرش": "sprinkler irrigation",
    "أخرى": "another method",

    "مياه النيل أو الترع": "Nile or canal water",
    "مياه الآبار": "well water",
    "مياه معالجة": "treated water",
    "مصدر آخر": "another source",

}


def model_to_dict(model):

    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)

    return model.dict(exclude_none=True)


def L(lang, arabic, english):

    return arabic if lang == "ar" else english


def normalize_backend_lang(raw_lang):

    code = (raw_lang or "ar").strip().lower()
    return "ar" if code.startswith("ar") else "en"


def get_language(farm):

    return normalize_backend_lang(farm.language)


def label(lang, value):

    return value if lang == "ar" else LABELS_EN.get(value, value)


def crop_name(lang, crop):

    if lang == "ar":
        return crop

    profile = CROP_PROFILES.get(crop)

    return profile["en"] if profile else crop


def month_name(lang, month):

    return MONTHS[month - 1] if lang == "ar" else MONTHS_EN[month - 1]


def months_text(lang, months):

    separator = "، " if lang == "ar" else ", "

    return separator.join(month_name(lang, m) for m in sorted(months))


def month_distance(first, second):

    difference = abs(first - second) % 12

    return min(difference, 12 - difference)


def window_distance(month, window):

    return min(month_distance(month, m) for m in window)


def arabic_lam(name):

    if name.startswith("ال"):
        return "لل" + name[2:]

    return "ل" + name


def gov_label(lang, governorate):

    return governorate if lang == "ar" else GOVERNORATE_EN.get(governorate, governorate)


def join_governorates(lang, values):

    separator = "، " if lang == "ar" else ", "

    return separator.join(gov_label(lang, v) for v in values)


FACTOR_NAMES = {
    "temperature": ("الحرارة", "temperature"),
    "soil": ("التربة", "soil type"),
    "irrigation": ("طريقة الري", "irrigation method"),
}


def join_labels(lang, values):

    separator = "، " if lang == "ar" else ", "

    return separator.join(label(lang, v) for v in values)


def farm_area_in_feddan(farm):

    if farm.farm_area is None:
        return None

    if farm.farm_area_unit == "هكتار":
        return round(farm.farm_area * HECTARE_TO_FEDDAN, 3)

    return round(float(farm.farm_area), 3)


# ============================================================
# 18. CLIMATE CONTEXT FOR THE EVALUATION
# ============================================================

def build_season(planting_month, weather_summary, climatology):

    if climatology:

        month = climatology.get(planting_month)

        if month and month["temp"] is not None:

            rain_day = month["rain_mm_day"]

            return {
                "source": "climatology",
                "temp": round(month["temp"], 1),
                "rain_total": (
                    round(rain_day * DAYS_IN_MONTH[planting_month - 1], 1)
                    if rain_day is not None else None
                ),
                "humidity": (
                    round(month["rh"], 1) if month["rh"] is not None else None
                ),
                "wind": (
                    round(month["ws"], 1) if month["ws"] is not None else None
                ),
                "solar": (
                    round(month["solar"], 1) if month.get("solar") is not None else None
                ),
                "period_ar": "شهر الزراعة حسب متوسط سنوات ناسا",
                "period_en": "the planting month per NASA's multi-year average",
            }

    if month_distance(planting_month, date.today().month) <= 1:

        return {
            "source": "recent",
            "temp": weather_summary["average_temperature_c"],
            "rain_total": weather_summary["total_rainfall_mm"],
            "humidity": weather_summary["average_relative_humidity_percent"],
            "wind": weather_summary["average_wind_speed_m_s"],
            "solar": weather_summary.get("average_solar_radiation_mj_m2_day"),
            "period_ar": "آخر 30 يومًا",
            "period_en": "the last 30 days",
        }

    return None


def build_climate_context(weather_summary, period_end, climatology, planting_month, season=None, soil_profile=None, elevation_data=None):

    resolved_season = (
        season
        if season is not None
        else build_season(planting_month, weather_summary, climatology)
    )

    return {
        "season": resolved_season,
        "period_end": period_end,
        "weather_summary": weather_summary,
        "soil_profile": soil_profile,
        "elevation_data": elevation_data,
    }


def season_text(lang, season):

    if not season or season["source"] != "climatology":

        return None

    parts_ar = [f"حرارة {season['temp']}°م"]

    parts_en = [f"temperature {season['temp']}°C"]

    if season["rain_total"] is not None:

        parts_ar.append(f"أمطار {season['rain_total']} مم في الشهر")

        parts_en.append(f"rain {season['rain_total']} mm per month")

    if season["humidity"] is not None:

        parts_ar.append(f"رطوبة {season['humidity']}%")

        parts_en.append(f"humidity {season['humidity']}%")

    if season["wind"] is not None:

        parts_ar.append(f"رياح {season['wind']} م/ث")

        parts_en.append(f"wind {season['wind']} m/s")

    return L(
        lang,
        "ℹ مناخ شهر الزراعة حسب ناسا (متوسط سنوات): " + "، ".join(parts_ar) + ".",
        "ℹ Planting-month climate from NASA (multi-year average): " + ", ".join(parts_en) + "."
    )


# ============================================================
# 19. PLANTING TIMING ALERT & ONE-CROP EVALUATION
# ============================================================

def build_planting_timing_alert(crop, planting_month, lang):

    if crop == "مش متأكد" or crop not in CROP_PROFILES:
        return None

    profile = CROP_PROFILES[crop]
    ideal_months = sorted(profile["months"])
    name = crop_name(lang, crop)
    chosen_name = month_name(lang, planting_month)
    ideal_str = months_text(lang, ideal_months)

    if planting_month in ideal_months:
        return {
            "status": "on_time",
            "planting_month": planting_month,
            "ideal_months": ideal_months,
            "nearest_ideal_month": planting_month,
            "months_difference": 0,
            "days_difference": 0,
            "recommended_action": "plant_now",
            "message": L(
                lang,
                f"ميعاد الزراعة المختار ({chosen_name}) مثالي {arabic_lam(name)}؛ يمكنك الزراعة في هذا الموعد.",
                f"Selected planting month ({chosen_name}) is ideal for {name}; you can plant on schedule."
            ),
        }

    forward_candidates = [
        ((target - planting_month) % 12, target)
        for target in ideal_months
    ]
    backward_candidates = [
        ((planting_month - target) % 12, target)
        for target in ideal_months
    ]

    forward_dist, forward_target = min(forward_candidates, key=lambda x: x[0])
    backward_dist, backward_target = min(backward_candidates, key=lambda x: x[0])

    if forward_dist <= backward_dist:
        months_diff = forward_dist
        days_diff = months_diff * 30
        target_name = month_name(lang, forward_target)
        return {
            "status": "early",
            "planting_month": planting_month,
            "ideal_months": ideal_months,
            "nearest_ideal_month": forward_target,
            "months_difference": months_diff,
            "days_difference": days_diff,
            "recommended_action": "wait",
            "message": L(
                lang,
                f"⏳ تنبيه ميعاد مبكر: شهر {chosen_name} يسبق الميعاد المثالي {arabic_lam(name)} ({ideal_str}) بنحو {months_diff} شهر (≈ {days_diff} يومًا). يُنصح بالانتظار نحو {days_diff} يومًا حتى شهر {target_name}.",
                f"⏳ Early planting alert: {chosen_name} is about {months_diff} month(s) (~{days_diff} days) earlier than the ideal window for {name} ({ideal_str}). Waiting ~{days_diff} days until {target_name} is recommended."
            ),
        }

    months_diff = backward_dist
    days_diff = months_diff * 30
    target_name = month_name(lang, backward_target)
    return {
        "status": "late",
        "planting_month": planting_month,
        "ideal_months": ideal_months,
        "nearest_ideal_month": backward_target,
        "months_difference": months_diff,
        "days_difference": days_diff,
        "recommended_action": "plant_earlier_or_next_season",
        "message": L(
            lang,
            f"⏳ تنبيه ميعاد متأخر: شهر {chosen_name} يتأخر عن الميعاد المثالي {arabic_lam(name)} ({ideal_str}) بنحو {months_diff} شهر (≈ {days_diff} يومًا) بعد شهر {target_name}. يُنصح بالتبكير بنحو {days_diff} يومًا في الموسم القادم أو اختيار صنف مبكر النضج.",
            f"⏳ Late planting alert: {chosen_name} is about {months_diff} month(s) (~{days_diff} days) later than the ideal window for {name} ({ideal_str}, after {target_name}). Planting ~{days_diff} days earlier next season or using an early-maturing variety is advised."
        ),
    }


STATUS_VALUE = {"good": 1.0, "warning": 0.5, "bad": 0.0}

CORE_FACTORS = ("planting_month", "temperature", "soil")


def evaluate_crop(crop, farm, ctx, lang):

    profile = CROP_PROFILES[crop]

    name = crop_name(lang, crop)

    ln = arabic_lam(name)

    cn = f"محصول {name}"

    checks = []

    def add(factor, status, weight, message):

        checks.append({
            "factor": factor,
            "status": status,
            "weight": weight,
            "message": message,
        })

    month = farm.planting_month

    season = ctx["season"]

    soil_profile = ctx.get("soil_profile") or {}
    elevation_data = ctx.get("elevation_data") or {}

    # ---- Planting month ----

    distance = window_distance(month, profile["months"])

    window = months_text(lang, profile["months"])

    planting = month_name(lang, month)

    if distance == 0:

        add("planting_month", "good", 3, L(
            lang,
            f"شهر الزراعة ({planting}) ضمن الموسم المعتاد {ln} في مصر: {window}.",
            f"Planting month ({planting}) is within the usual season for {name} in Egypt: {window}."
        ))

    elif distance == 1:

        add("planting_month", "warning", 3, L(
            lang,
            f"شهر الزراعة ({planting}) قريب من الموسم المعتاد {ln} ({window}) لكنه ليس الأفضل.",
            f"Planting month ({planting}) is close to the usual season for {name} ({window}) but not ideal."
        ))

    else:

        add("planting_month", "bad", 3, L(
            lang,
            f"شهر الزراعة ({planting}) خارج الموسم المعتاد {ln} في مصر. الأنسب: {window}.",
            f"Planting month ({planting}) is outside the usual season for {name} in Egypt. Best: {window}."
        ))

    # ---- Temperature ----

    low, high = profile["temp_opt"]

    abs_low, abs_high = profile["temp_abs"]

    if season is not None:

        temperature = season["temp"]

        period_ar = season["period_ar"]

        period_en = season["period_en"]

        if low <= temperature <= high:

            add("temperature", "good", 2, L(
                lang,
                f"متوسط الحرارة في {period_ar} ({temperature}°م) داخل المدى الأمثل {ln} ({low}–{high}°م).",
                f"The mean temperature in {period_en} ({temperature}°C) is within the optimum range for {name} ({low}-{high}°C)."
            ))

        elif abs_low <= temperature <= abs_high:

            add("temperature", "warning", 2, L(
                lang,
                f"متوسط الحرارة في {period_ar} ({temperature}°م) خارج المدى الأمثل {ln} ({low}–{high}°م) لكنه ضمن المدى المحتمل.",
                f"The mean temperature in {period_en} ({temperature}°C) is outside the optimum range for {name} ({low}-{high}°C) but within the tolerable range."
            ))

        else:

            add("temperature", "bad", 2, L(
                lang,
                f"متوسط الحرارة في {period_ar} ({temperature}°م) خارج المدى المحتمل {ln} ({abs_low}–{abs_high}°م).",
                f"The mean temperature in {period_en} ({temperature}°C) is outside the tolerable range for {name} ({abs_low}-{abs_high}°C)."
            ))

    else:

        add("temperature", "info", 0, L(
            lang,
            f"تعذر الحصول على مناخ شهر الزراعة من ناسا، لذلك لم يُحكم على الحرارة. المدى الأمثل {ln}: {low}–{high}°م.",
            f"NASA climate for the planting month could not be retrieved, so temperature was not judged. Optimum range for {name}: {low}-{high}°C."
        ))

    # ---- Soil (Hybrid SoilGrids + Manual Lab + Farmer Selection) ----

    soil = soil_profile.get("effective_soil_type") or farm.soil_type

    soil_text = label(lang, soil)

    if soil == "لا أعرف":

        add("soil", "info", 0, L(
            lang,
            "نوع التربة غير معروف، لذلك لم يُحكم على ملاءمتها. حدّد نوع التربة أو اعمل تحليلًا لها.",
            "Soil type is unknown, so soil suitability was not judged. Identify the soil type or test it."
        ))

    else:

        rating = profile["soil"][soil]

        inferred_note_ar = ""
        inferred_note_en = ""
        if farm.soil_type == "لا أعرف" and soil_profile.get("inferred_texture_from_data"):
            inferred_note_ar = f" (مقدّرة من {soil_profile.get('source_label', 'SoilGrids')})"
            inferred_note_en = f" (inferred from {soil_profile.get('source_label', 'SoilGrids')})"

        if rating == "p":

            add("soil", "good", 3, L(
                lang,
                f"التربة {soil_text}{inferred_note_ar} مناسبة {ln}.",
                f"{soil_text.capitalize()} soil{inferred_note_en} suits {name}."
            ))

        elif rating == "t":

            add("soil", "warning", 3, L(
                lang,
                f"التربة {soil_text}{inferred_note_ar} تصلح {ln} لكن بشروط (إدارة الري والتسميد والصرف).",
                f"{soil_text.capitalize()} soil{inferred_note_en} can work for {name} but needs careful irrigation, fertilizer and drainage management."
            ))

        else:

            add("soil", "bad", 3, L(
                lang,
                f"التربة {soil_text}{inferred_note_ar} غير مناسبة عادةً {ln}.",
                f"{soil_text.capitalize()} soil{inferred_note_en} is usually unsuitable for {name}."
            ))

    # ---- Hybrid Soil pH Check ----

    soil_ph = soil_profile.get("ph")
    if soil_ph is not None:
        ph_min, ph_max = profile["ph_range"]
        if ph_min <= soil_ph <= ph_max:
            add("soil_ph", "good", 0.5, L(
                lang,
                f"درجة حموضة التربة (pH = {soil_ph}) ضمن النطاق المناسب {ln} ({ph_min}–{ph_max}).",
                f"Soil pH ({soil_ph}) is within the suitable range for {name} ({ph_min}-{ph_max})."
            ))
        else:
            add("soil_ph", "warning", 0.5, L(
                lang,
                f"درجة حموضة التربة (pH = {soil_ph}) خارج النطاق المفضل {ln} ({ph_min}–{ph_max})؛ قد تحتاج تعديلاً بالجبس الزراعي أو الكبريت.",
                f"Soil pH ({soil_ph}) is outside the preferred range for {name} ({ph_min}-{ph_max}); gypsum or elemental sulfur may be needed."
            ))

    # ---- Standing water ----

    standing = farm.standing_water

    tolerance = profile["waterlog"]

    if standing == "لا":

        add("drainage", "good", 2, L(
            lang,
            "لا تتجمع مياه بعد الري، وهذا جيد.",
            "No standing water after irrigation, which is good."
        ))

    elif tolerance == "tolerant":

        add("drainage", "good", 2, L(
            lang,
            f"{cn} يتحمل وجود الماء الراكد.",
            f"{name} tolerates standing water."
        ))

    elif standing == "نعم" and tolerance == "medium":

        add("drainage", "warning", 2, L(
            lang,
            f"تجمع المياه بعد الري مشكلة متوسطة {ln}؛ حسّن الصرف.",
            f"Standing water after irrigation is a moderate problem for {name}; improve drainage."
        ))

    elif standing == "نعم":

        add("drainage", "bad", 2, L(
            lang,
            f"تجمع المياه بعد الري خطر كبير على {name}؛ حسّن الصرف قبل الزراعة.",
            f"Standing water after irrigation is a serious risk for {name}; improve drainage before planting."
        ))

    elif tolerance == "medium":

        add("drainage", "good", 2, L(
            lang,
            f"تجمع المياه أحيانًا مقبول نسبيًا {ln}.",
            f"Occasional standing water is fairly acceptable for {name}."
        ))

    else:

        add("drainage", "warning", 2, L(
            lang,
            f"تجمع المياه أحيانًا قد يضر {name}؛ راقب الصرف.",
            f"Occasional standing water can harm {name}; monitor drainage."
        ))

    # ---- Irrigation method & SRTM Elevation Efficiency ----

    method = farm.irrigation_method

    best_methods = profile["irrigation"]

    method_text = label(lang, method)

    if method == "أخرى":

        add("irrigation", "info", 0, L(
            lang,
            "طريقة الري المذكورة (أخرى) غير محددة، لذلك لم يُحكم عليها.",
            "The irrigation method (other) is not specific, so it was not judged."
        ))

    elif method in best_methods:

        add("irrigation", "good", 1.5, L(
            lang,
            f"طريقة الري ({method_text}) مناسبة {ln}.",
            f"Irrigation method ({method_text}) suits {name}."
        ))

    else:

        add("irrigation", "warning", 1.5, L(
            lang,
            f"طريقة الري ({method_text}) ليست الأنسب {ln}. الأفضل: {join_labels(lang, best_methods)}.",
            f"Irrigation method ({method_text}) is not the best for {name}. Better: {join_labels(lang, best_methods)}."
        ))

    # ---- SRTM Elevation check when Flood irrigation is used ----

    elev_m = elevation_data.get("elevation_m")
    if elev_m is not None and method == "الغمر" and crop != "الأرز":
        if elev_m > 80.0 or elev_m <= 15.0:
            add("elevation_irrigation", "warning", 0.5, L(
                lang,
                elevation_data.get("flood_vs_drip_note_ar", ""),
                elevation_data.get("flood_vs_drip_note_en", ""),
            ))

    # ---- Irrigation method together with soil ----

    if method == "الغمر" and soil == "رملية" and crop != "الأرز":

        add("irrigation_soil", "warning", 0.5, L(
            lang,
            "الري بالغمر في التربة الرملية يضيّع الماء والأسمدة بالرشح.",
            "Flood irrigation on sandy soil wastes water and fertilizer through leaching."
        ))

    elif method == "الغمر" and soil == "طينية" and tolerance in ("low", "very_low"):

        add("irrigation_soil", "warning", 0.5, L(
            lang,
            f"الري بالغمر في التربة الطينية يزيد خطر تشبع التربة، و{cn} حساس لذلك.",
            f"Flood irrigation on clay soil increases waterlogging risk, and {name} is sensitive to it."
        ))

    # ---- Water source ----

    source = farm.water_source

    source_text = label(lang, source)

    if source == "مياه النيل أو الترع":

        add("water_source", "good", 1, L(
            lang,
            "مياه النيل أو الترع مصدر مناسب عمومًا.",
            "Nile or canal water is generally a suitable source."
        ))

    elif source == "مياه معالجة" and profile["fresh"]:

        add("water_source", "warning", 1, L(
            lang,
            f"المياه المعالجة قد لا تناسب {name} لأن المحصول يُؤكل طازجًا؛ تأكد من درجة المعالجة المسموح بها صحيًا.",
            f"Treated water may not suit {name} because it is eaten fresh; confirm the treatment grade is permitted for food crops."
        ))

    elif source == "مياه الآبار" and profile["salt"] == "low":

        add("water_source", "warning", 1, L(
            lang,
            f"مياه الآبار قد تكون مرتفعة الملوحة، و{cn} حساس للملوحة؛ حلّل ملوحة المياه.",
            f"Well water can be saline and {name} is salt-sensitive; test the water salinity."
        ))

    else:

        add("water_source", "info", 0, L(
            lang,
            f"مصدر المياه ({source_text}): يُفضّل عمل تحليل لجودة المياه.",
            f"Water source ({source_text}): a water quality test is recommended."
        ))

    # ---- Salinity reported by the farmer OR manual lab EC ----

    manual_ec = soil_profile.get("ec_ds_m")
    if farm.agricultural_problem == "ملوحة التربة" or (manual_ec is not None and manual_ec >= 2.5):

        salt = profile["salt"]

        ec_note_ar = f" (EC = {manual_ec} dS/m)" if manual_ec is not None else ""
        ec_note_en = f" (EC = {manual_ec} dS/m)" if manual_ec is not None else ""

        if salt == "low":

            add("salinity", "bad", 1, L(
                lang,
                f"ملوحة التربة مرتفعة{ec_note_ar}، و{cn} حساس للملوحة.",
                f"Soil salinity is elevated{ec_note_en} and {name} is salt-sensitive."
            ))

        elif salt == "medium":

            add("salinity", "warning", 1, L(
                lang,
                f"ملوحة التربة ملحوظة{ec_note_ar}، و{cn} يتحمل الملوحة بدرجة متوسطة فقط.",
                f"Soil salinity is notable{ec_note_en} and {name} has only moderate salt tolerance."
            ))

        else:

            add("salinity", "good", 1, L(
                lang,
                f"{name} من المحاصيل التي تتحمل الملوحة نسبيًا{ec_note_ar}.",
                f"{name} is relatively salt-tolerant{ec_note_en}."
            ))

    # ---- Humidity, rain and wind ----

    if season is not None:

        period_ar = season["period_ar"]

        period_en = season["period_en"]

        humidity = season["humidity"]

        if humidity is not None and profile["humid_limit"] is not None:

            if humidity > profile["humid_limit"]:

                add("humidity", "warning", 1, L(
                    lang,
                    f"الرطوبة النسبية في {period_ar} ({humidity}%) مرتفعة وتزيد خطر الأمراض الفطرية في {name}.",
                    f"Relative humidity in {period_en} ({humidity}%) is high and raises fungal disease risk in {name}."
                ))

            else:

                add("humidity", "good", 1, L(
                    lang,
                    f"الرطوبة النسبية في {period_ar} ({humidity}%) لا تثير قلقًا كبيرًا {ln}.",
                    f"Relative humidity in {period_en} ({humidity}%) is not a major concern for {name}."
                ))

        rain = season["rain_total"]

        if rain is not None and rain > 40 and tolerance in ("low", "very_low"):

            add("rain", "warning", 0.5, L(
                lang,
                f"أمطار {period_ar} ({rain} مم) مرتفعة، و{cn} لا يتحمل تشبع التربة.",
                f"Rainfall in {period_en} ({rain} mm) is high and {name} does not tolerate waterlogging."
            ))

        wind = season["wind"]

        if profile["wind_sensitive"] and wind is not None and wind > 5:

            add("wind", "warning", 0.5, L(
                lang,
                f"متوسط سرعة الرياح في {period_ar} ({wind} م/ث) مرتفع و{cn} حساس للرياح؛ استخدم مصدات الرياح.",
                f"Average wind speed in {period_en} ({wind} m/s) is high and {name} is wind-sensitive; use windbreaks."
            ))

    # ---- Governorate: typical growing areas ----

    rule = REGION_RULES.get(crop)

    gov = gov_label(lang, farm.governorate)

    if rule:

        if farm.governorate in rule["main"]:

            add("region", "good", rule["weight"], L(
                lang,
                rule["ar_ok"].format(name=name, gov=gov),
                rule["en_ok"].format(name=name, gov=gov)
            ))

        else:

            regions = join_governorates(lang, rule["main"])

            add("region", "warning", rule["weight"], L(
                lang,
                rule["ar_bad"].format(name=name, gov=gov, regions=regions),
                rule["en_bad"].format(name=name, gov=gov, regions=regions)
            ))

    # ---- Governorate: very hot summers ----

    if (
        farm.governorate in HOT_SUMMER_GOVERNORATES
        and month in (5, 6, 7, 8, 9)
        and distance == 0
        and profile["temp_abs"][1] <= 35
    ):

        add("regional_heat", "warning", 1, L(
            lang,
            f"صيف {gov} شديد الحرارة، و{cn} يتأثر بالحرارة العالية؛ راعِ ميعاد الزراعة والري.",
            f"Summers in {gov} are very hot and {name} is affected by high heat; mind the planting date and irrigation."
        ))

    # ---- Score and overall suitability ----

    scored = [c for c in checks if c["status"] != "info"]

    total_weight = sum(c["weight"] for c in scored)

    if total_weight > 0:
        score = round(
            sum(STATUS_VALUE[c["status"]] * c["weight"] for c in scored)
            / total_weight * 100
        )
    else:
        score = 0

    if rule and farm.governorate not in rule["main"]:

        score = min(score, 60 if rule["weight"] >= 2 else 75)

    core_bad = any(
        c["factor"] in CORE_FACTORS and c["status"] == "bad"
        for c in checks
    )

    any_problem = any(c["status"] in ("bad", "warning") for c in checks)

    if core_bad:
        suitability = "not_suitable"
    elif any_problem:
        suitability = "suitable_with_conditions"
    else:
        suitability = "suitable"

    core_weights = {
        "planting_month": 3, "temperature": 2, "soil": 3,
        "drainage": 2, "irrigation": 1.5,
    }

    unjudged = [
        factor for factor in core_weights
        if not any(
            c["factor"] == factor and c["status"] != "info" for c in checks
        )
    ]

    judged_weight = sum(
        weight for factor, weight in core_weights.items()
        if factor not in unjudged
    )

    ratio = judged_weight / sum(core_weights.values())

    if ratio >= 0.85:
        confidence = "high"
    elif ratio >= 0.65:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "crop": crop,
        "name": name,
        "score": score,
        "suitability": suitability,
        "confidence": confidence,
        "unjudged": unjudged,
        "checks": checks,
        "scored_weight_sum": round(total_weight, 2),
        "weighted_points": round(sum(STATUS_VALUE[c["status"]] * c["weight"] for c in scored), 2),
    }


def suitability_label(lang, code):

    labels = {
        "suitable": ("مناسب", "Suitable"),
        "suitable_with_conditions": ("مناسب بشروط", "Suitable with conditions"),
        "not_suitable": ("غير مناسب بالشكل الحالي", "Not suitable as currently planned"),
    }

    arabic, english = labels[code]

    return L(lang, arabic, english)


# ============================================================
# 20. IRRIGATION RECOMMENDATION & LIVE NASA WATER DEMAND MODEL
# ============================================================

def best_irrigation_method_for_crop(crop, farm, soil_profile=None, elevation_data=None):

    profile = CROP_PROFILES[crop]
    options = list(profile["irrigation"])
    effective_soil = (soil_profile or {}).get("effective_soil_type") or farm.soil_type
    elev_m = (elevation_data or {}).get("elevation_m")

    if crop != "الأرز":

        if effective_soil == "رملية":
            options = [o for o in options if o != "الغمر"] or options

        if (
            effective_soil == "طينية"
            and profile["waterlog"] in ("low", "very_low")
            and "التنقيط" in options
        ):
            options = ["التنقيط"] + [o for o in options if o != "التنقيط"]

        if elev_m is not None and (elev_m > 80.0 or elev_m <= 15.0) and "التنقيط" in options:
            options = ["التنقيط"] + [o for o in options if o != "التنقيط"]

    return options[0], options[1:]


def irrigation_recommendation(crop, farm, lang, soil_profile=None, elevation_data=None):

    name = crop_name(lang, crop)

    ln = arabic_lam(name)

    best, others = best_irrigation_method_for_crop(
        crop, farm, soil_profile=soil_profile, elevation_data=elevation_data
    )

    if crop == "الأرز":

        reason_ar = "الأرز يحتاج غمرًا بطبقة ماء، وهي الطريقة المناسبة له."

        reason_en = "Rice needs a flooded water layer, which is the suitable method for it."

    else:

        reason_ar, reason_en = IRRIGATION_REASONS[best]

    best_text = label(lang, best)

    text = L(
        lang,
        f"أنسب طريقة ري {ln} في ظروفك: {best_text}. {reason_ar}",
        f"Most suitable irrigation method for {name} in your conditions: {best_text}. {reason_en}"
    )

    if others:

        text += L(
            lang,
            f" بدائل مقبولة: {join_labels(lang, others)}.",
            f" Acceptable alternatives: {join_labels(lang, others)}."
        )

    return text


def calculate_live_crop_water_demand(
    crop, farm, weather_summary, season, forecast_days, soil_profile, elevation_data, lang
):
    """
    Live dynamic water demand calculation (m3 / feddan) combining:
      1) NASA POWER SMAP/MERRA-2 soil moisture (GWETTOP, GWETROOT)
      2) NASA POWER EVPTRNS & Solar Radiation (ALLSKY_SFC_SW_DWN)
      3) Open-Meteo FAO ET0 (et0_fao_evapotranspiration)
      4) Crop Coefficient (Kc), Season Duration, SRTM Elevation & Soil Texture
    """
    if crop == "مش متأكد" or crop not in CROP_ECONOMICS or crop not in CROP_PROFILES:
        return None

    econ = CROP_ECONOMICS[crop]
    profile = CROP_PROFILES[crop]
    duration_days = econ["duration_days"]
    kc = profile["kc"]

    open_meteo_et0_vals = [
        d["et0_mm"] for d in (forecast_days or [])
        if d.get("et0_mm") is not None
    ]
    forecast_et0_avg = round(mean(open_meteo_et0_vals), 2) if open_meteo_et0_vals else None
    nasa_daily_et = weather_summary.get("estimated_daily_et_mm")

    if forecast_et0_avg is not None and nasa_daily_et is not None:
        live_et0_mm_day = round(0.60 * forecast_et0_avg + 0.40 * nasa_daily_et, 2)
    elif forecast_et0_avg is not None:
        live_et0_mm_day = forecast_et0_avg
    elif nasa_daily_et is not None:
        live_et0_mm_day = nasa_daily_et
    else:
        live_et0_mm_day = 4.2

    if season and season.get("temp") is not None:
        season_temp = season["temp"]
        season_solar = season.get("solar") or weather_summary.get("average_solar_radiation_mj_m2_day") or 19.0
        seasonal_et0_est = max(2.2, 0.0135 * (season_temp + 17.78) * (season_solar * 0.408))
        blended_et0_mm_day = round(0.45 * live_et0_mm_day + 0.55 * seasonal_et0_est, 2)
    else:
        blended_et0_mm_day = live_et0_mm_day

    monthly_rain = (season or {}).get("rain_total")
    if monthly_rain is None:
        monthly_rain = weather_summary.get("total_rainfall_mm") or 0.0
    effective_seasonal_rain_mm = round(min(180.0, monthly_rain * (duration_days / 30.0) * 0.65), 1)

    gwet_top = weather_summary.get("average_surface_soil_wetness")
    gwet_root = weather_summary.get("average_root_zone_soil_wetness")
    effective_soil = (soil_profile or {}).get("effective_soil_type") or farm.soil_type

    smap_credit_mm = 0.0
    soil_moisture_status = "normal"
    if gwet_root is not None:
        if gwet_root >= 0.60:
            smap_credit_mm = 45.0
            soil_moisture_status = "wet"
        elif gwet_root >= 0.42:
            smap_credit_mm = 22.0
            soil_moisture_status = "moderate"
        elif gwet_root < 0.25:
            smap_credit_mm = -25.0
            soil_moisture_status = "dry"

    crop_etc_mm = blended_et0_mm_day * kc * duration_days
    net_irrigation_mm = max(220.0, crop_etc_mm - effective_seasonal_rain_mm - smap_credit_mm)

    rice_puddling_mm = 480.0 if crop == "الأرز" else 0.0
    net_m3_per_feddan = (net_irrigation_mm + rice_puddling_mm) * (FEDDAN_AREA_M2 / 1000.0)

    soil_loss_factor = 1.15 if effective_soil == "رملية" else (0.98 if effective_soil == "طينية" else 1.03)
    elev_m = (elevation_data or {}).get("elevation_m") or 25.0
    elev_flood_penalty = 1.08 if elev_m > 80.0 else (1.05 if elev_m <= 15.0 else 1.0)

    flood_m3_per_feddan = round(
        (net_m3_per_feddan / IRRIGATION_EFFICIENCY["الغمر"]) * soil_loss_factor * elev_flood_penalty
    )
    baseline_flood = econ["water_flood_m3"]
    flood_m3_per_feddan = max(round(baseline_flood * 0.72), min(round(baseline_flood * 1.35), flood_m3_per_feddan))

    best_method, _ = best_irrigation_method_for_crop(
        crop, farm, soil_profile=soil_profile, elevation_data=elevation_data
    )

    if crop == "الأرز":
        saving_ratio = RICE_AWD_SAVING_VS_FLOOD
        recommended_m3_per_feddan = round(flood_m3_per_feddan * (1.0 - saving_ratio))
    else:
        eff_best = IRRIGATION_EFFICIENCY.get(best_method, 0.85)
        eff_flood = IRRIGATION_EFFICIENCY["الغمر"]
        recommended_m3_per_feddan = round(flood_m3_per_feddan * (eff_flood / eff_best))
        recommended_m3_per_feddan = min(flood_m3_per_feddan, recommended_m3_per_feddan)

    eff_current = IRRIGATION_EFFICIENCY.get(farm.irrigation_method, 0.65)
    if crop == "الأرز" or farm.irrigation_method == "الغمر":
        current_m3_per_feddan = flood_m3_per_feddan
    else:
        current_m3_per_feddan = round(flood_m3_per_feddan * (IRRIGATION_EFFICIENCY["الغمر"] / eff_current))

    daily_net_m3_per_feddan = round(recommended_m3_per_feddan / max(1, duration_days), 2)

    return {
        "crop": crop,
        "kc": kc,
        "duration_days": duration_days,
        "live_nasa_et_mm_day": nasa_daily_et,
        "open_meteo_et0_mm_day": forecast_et0_avg,
        "blended_et0_mm_day": blended_et0_mm_day,
        "smap_surface_wetness_gwettop": gwet_top,
        "smap_root_zone_wetness_gwetroot": gwet_root,
        "soil_moisture_status": soil_moisture_status,
        "effective_rainfall_mm": effective_seasonal_rain_mm,
        "elevation_m": elev_m,
        "flood_m3_per_feddan": flood_m3_per_feddan,
        "current_method_m3_per_feddan": current_m3_per_feddan,
        "recommended_method": best_method,
        "recommended_m3_per_feddan": recommended_m3_per_feddan,
        "daily_avg_m3_per_feddan": daily_net_m3_per_feddan,
        "calculation_basis": L(
            lang,
            "محسوب حياً من رطوبة الجذور (NASA SMAP) والبخر-نتح (EVPTRNS & Open-Meteo FAO ET0) وارتفاع الأرض (SRTM)",
            "Calculated live from NASA SMAP root-zone moisture, EVPTRNS & Open-Meteo FAO ET0, and SRTM elevation"
        ),
    }


# ============================================================
# 20B. DYNAMIC ROI & WATER SAVINGS CALCULATOR
# ============================================================

def calculate_economics_and_water(
    crop, farm, lang,
    live_water=None,
    market_indicators=None,
    apply_manual_overrides=True
):

    if crop == "مش متأكد" or crop not in CROP_ECONOMICS:
        return None

    econ = CROP_ECONOMICS[crop]
    area_feddan = farm_area_in_feddan(farm)
    effective_area = area_feddan if area_feddan is not None else 1.0

    mkt = market_indicators or {
        "usd_egp_rate": FALLBACK_USD_EGP,
        "cost_index_factor": 1.0,
        "price_index_factor": 1.0,
        "source": "Baseline Market Fallback",
    }

    dynamic_cost = round(econ["cost"] * mkt["cost_index_factor"], -2)
    dynamic_price = round(econ["price_per_ton"] * mkt["price_index_factor"], -1)
    dynamic_yield = econ["yield_ton"]

    manual_override_used = False
    if apply_manual_overrides:
        if farm.custom_cost_per_feddan is not None:
            dynamic_cost = round(float(farm.custom_cost_per_feddan), 2)
            manual_override_used = True
        if farm.custom_price_per_ton is not None:
            dynamic_price = round(float(farm.custom_price_per_ton), 2)
            manual_override_used = True
        if farm.custom_yield_ton_per_feddan is not None:
            dynamic_yield = round(float(farm.custom_yield_ton_per_feddan), 2)
            manual_override_used = True

    cost_per_feddan = dynamic_cost
    yield_ton = dynamic_yield
    price_per_ton = dynamic_price
    gross_revenue_per_feddan = round(yield_ton * price_per_ton)
    net_profit_per_feddan = round(gross_revenue_per_feddan - cost_per_feddan)
    roi_percent = round((net_profit_per_feddan / cost_per_feddan) * 100, 1) if cost_per_feddan > 0 else 0.0

    if live_water is not None:
        baseline_water_m3 = live_water["flood_m3_per_feddan"]
        current_water_m3 = live_water["current_method_m3_per_feddan"]
        recommended_water_m3 = live_water["recommended_m3_per_feddan"]
        best_method = live_water["recommended_method"]
    else:
        best_method, _ = best_irrigation_method_for_crop(crop, farm)
        baseline_water_m3 = econ["water_flood_m3"]
        saving_ratio_fb = RICE_AWD_SAVING_VS_FLOOD if crop == "الأرز" else IRRIGATION_SAVING_VS_FLOOD.get(best_method, 0.0)
        current_ratio_fb = 0.0 if crop == "الأرز" else IRRIGATION_SAVING_VS_FLOOD.get(farm.irrigation_method, 0.0)
        recommended_water_m3 = round(baseline_water_m3 * (1 - saving_ratio_fb))
        current_water_m3 = round(baseline_water_m3 * (1 - current_ratio_fb))

    if crop == "الأرز":
        recommended_method_label = L(
            lang,
            "الغمر مع الري المتقطع (AWD)",
            "Flood with Alternate Wetting and Drying (AWD)"
        )
    else:
        recommended_method_label = label(lang, best_method)

    water_saved_vs_flood_m3 = max(0, round(baseline_water_m3 - recommended_water_m3))
    water_saved_vs_current_m3 = max(0, round(current_water_m3 - recommended_water_m3))
    water_saved_percent = (
        round((water_saved_vs_flood_m3 / baseline_water_m3) * 100, 1)
        if baseline_water_m3 > 0 else 0.0
    )

    return {
        "crop": crop,
        "crop_en": CROP_PROFILES[crop]["en"],
        "currency": "EGP",
        "area_feddan": round(effective_area, 3),
        "is_area_user_specified": area_feddan is not None,
        "season_duration_days": econ["duration_days"],
        "market_indexing": {
            "usd_egp_rate": mkt.get("usd_egp_rate"),
            "cost_index_factor": mkt.get("cost_index_factor"),
            "price_index_factor": mkt.get("price_index_factor"),
            "manual_override_applied": manual_override_used,
            "source": mkt.get("source"),
        },
        "economics": {
            "cost_per_feddan_egp": cost_per_feddan,
            "expected_yield_ton_per_feddan": yield_ton,
            "price_per_ton_egp": price_per_ton,
            "gross_revenue_per_feddan_egp": gross_revenue_per_feddan,
            "net_profit_per_feddan_egp": net_profit_per_feddan,
            "roi_percent": roi_percent,
            "total_cost_egp": round(cost_per_feddan * effective_area),
            "total_gross_revenue_egp": round(gross_revenue_per_feddan * effective_area),
            "total_net_profit_egp": round(net_profit_per_feddan * effective_area),
        },
        "water_savings": {
            "calculation_mode": "live_nasa_open_meteo" if live_water is not None else "baseline",
            "traditional_flood_m3_per_feddan": baseline_water_m3,
            "current_method": label(lang, farm.irrigation_method),
            "current_method_m3_per_feddan": current_water_m3,
            "recommended_method": recommended_method_label,
            "recommended_method_m3_per_feddan": recommended_water_m3,
            "saved_vs_flood_m3_per_feddan": water_saved_vs_flood_m3,
            "saved_vs_current_m3_per_feddan": water_saved_vs_current_m3,
            "water_saved_percent_vs_flood": water_saved_percent,
            "total_water_saved_vs_flood_m3": round(water_saved_vs_flood_m3 * effective_area),
            "total_water_saved_vs_current_m3": round(water_saved_vs_current_m3 * effective_area),
        },
        "summary_text": L(
            lang,
            (
                f"التقدير الاقتصادي الديناميكي للفدان (بسعر صرف {mkt.get('usd_egp_rate')} ج.م/$): "
                f"تكلفة ≈ {cost_per_feddan:,} ج.م، عائد إجمالي ≈ {gross_revenue_per_feddan:,} ج.م، "
                f"صافي ربح متوقع ≈ {net_profit_per_feddan:,} ج.م (عائد على الاستثمار {roi_percent}%). "
                f"الاحتياج المائي الحي: يوفر نظام «{recommended_method_label}» نحو {water_saved_vs_flood_m3:,} م³/فدان ({water_saved_percent}%)."
            ),
            (
                f"Dynamic per-Feddan estimate (USD/EGP {mkt.get('usd_egp_rate')}): "
                f"cost ~{cost_per_feddan:,} EGP, gross revenue ~{gross_revenue_per_feddan:,} EGP, "
                f"expected net profit ~{net_profit_per_feddan:,} EGP (ROI {roi_percent}%). "
                f"Live water model: '{recommended_method_label}' saves ~{water_saved_vs_flood_m3:,} m³/feddan ({water_saved_percent}%)."
            ),
        ),
        "disclaimer": L(
            lang,
            "الأرقام الاقتصادية محدّثة بمؤشر سعر الصرف والأسمدة العالمي مع إمكانية التعديل اليدوي، والأرقام المائية محسوبة حياً من بيانات ناسا وOpen-Meteo.",
            "Economic figures are dynamically indexed to live USD/EGP & input indices (with manual override support); water demand is computed live from NASA & Open-Meteo."
        ),
    }


# ============================================================
# 20C. INTERACTIVE CROP CALENDAR & 16-DAY FAO ET0 SCHEDULE
# ============================================================

def _reference_planting_date(planting_month):

    today = date.today()
    year = today.year if planting_month >= today.month else today.year + 1
    return date(year, planting_month, 15)


def build_16_day_irrigation_schedule(crop, farm, forecast_days, weather_summary, soil_profile, lang):
    """
    Operational 16-day irrigation & fertilization schedule driven by:
      - Open-Meteo FAO Penman-Monteith ET0 (et0_fao_evapotranspiration)
      - Forecasted Rainfall
      - NASA SMAP Root-Zone Soil Wetness (GWETROOT)
      - Soil Texture & Irrigation Efficiency
    """
    if not forecast_days:
        return None

    profile = CROP_PROFILES.get(crop) if crop != "مش متأكد" else None
    kc = profile["kc"] if profile else 0.85
    eff_soil = (soil_profile or {}).get("effective_soil_type") or farm.soil_type
    eff_irr = IRRIGATION_EFFICIENCY.get(farm.irrigation_method, 0.70)
    gwet_root = weather_summary.get("average_root_zone_soil_wetness")

    if eff_soil == "رملية":
        raw_threshold_mm = 16.0
    elif eff_soil == "طينية":
        raw_threshold_mm = 32.0
    else:
        raw_threshold_mm = 24.0

    if gwet_root is not None and gwet_root >= 0.55:
        cumulative_deficit_mm = 0.0
    elif gwet_root is not None and gwet_root >= 0.35:
        cumulative_deficit_mm = raw_threshold_mm * 0.35
    else:
        cumulative_deficit_mm = raw_threshold_mm * 0.65

    schedule = []
    total_et0_16d = 0.0
    total_net_m3_feddan = 0.0

    for idx, d in enumerate(forecast_days[:FORECAST_DAYS], start=1):
        et0 = d.get("et0_mm") or 4.0
        rain = d.get("rain_mm") or 0.0
        wind = d.get("wind_max_kmh") or 0.0
        t_max = d.get("t_max") or 25.0

        crop_etc = round(et0 * kc, 2)
        effective_rain = round(rain * 0.80, 2)
        total_et0_16d += et0

        cumulative_deficit_mm = max(0.0, cumulative_deficit_mm + crop_etc - effective_rain)

        irrigate_today = False
        gross_irrigation_mm = 0.0
        if rain >= 5.0:
            action = "skip_rain"
            action_text = L(lang, "تأجيل الري (أمطار متوقعة)", "Skip irrigation (rain forecast)")
        elif cumulative_deficit_mm >= raw_threshold_mm:
            irrigate_today = True
            gross_irrigation_mm = round(cumulative_deficit_mm / eff_irr, 1)
            cumulative_deficit_mm = 0.0
            action = "irrigate"
            action_text = L(
                lang,
                f"موعد رية رئيسية (~{round(gross_irrigation_mm * 4.2, 1)} م³/فدان)",
                f"Irrigate (~{round(gross_irrigation_mm * 4.2, 1)} m³/feddan)"
            )
        elif farm.irrigation_method == "التنقيط" and idx % 2 == 1 and cumulative_deficit_mm >= raw_threshold_mm * 0.35:
            irrigate_today = True
            gross_irrigation_mm = round(cumulative_deficit_mm / eff_irr, 1)
            cumulative_deficit_mm = 0.0
            action = "drip_pulse"
            action_text = L(
                lang,
                f"رية تنقيط خفيفة (~{round(gross_irrigation_mm * 4.2, 1)} م³/فدان)",
                f"Light drip irrigation (~{round(gross_irrigation_mm * 4.2, 1)} m³/feddan)"
            )
        else:
            action = "monitor"
            action_text = L(lang, "لا يلزم ري اليوم — متابعة الرطوبة", "No irrigation needed — monitor moisture")

        gross_m3_feddan = round(gross_irrigation_mm * (FEDDAN_AREA_M2 / 1000.0), 1)
        total_net_m3_feddan += gross_m3_feddan

        if rain >= 5.0 or wind >= 35.0:
            fert_note = L(lang, "تأجيل الرش الورقي (أمطار أو رياح)", "Avoid foliar spray (rain/wind)")
        elif irrigate_today and t_max < 36.0:
            fert_note = L(lang, "مناسب للتسميد مع ماء الري (تسميد أرضي/نقطي)", "Suitable for fertigation / soil fertilization")
        elif t_max >= 36.0:
            fert_note = L(lang, "العمل صباحاً فقط (حرارة عالية)", "Morning field work only (high heat)")
        else:
            fert_note = L(lang, "يوم مناسب للعمليات الحقلية والرش الآمن", "Suitable for field operations & safe spraying")

        schedule.append({
            "day_index": idx,
            "date": d["date"],
            "et0_fao_mm": round(et0, 2),
            "crop_etc_mm": crop_etc,
            "rain_mm": round(rain, 1),
            "irrigate": irrigate_today,
            "action_code": action,
            "irrigation_action": action_text,
            "recommended_water_m3_per_feddan": gross_m3_feddan,
            "fertilization_and_spray_window": fert_note,
        })

    return {
        "source": "Open-Meteo FAO ET0 + NASA SMAP Soil Wetness",
        "crop_kc_used": kc,
        "smap_initial_root_wetness": gwet_root,
        "total_16d_et0_mm": round(total_et0_16d, 1),
        "total_16d_recommended_water_m3_per_feddan": round(total_net_m3_feddan, 1),
        "days": schedule,
    }


def build_crop_calendar(crop, farm, lang, soil_profile=None, live_water=None, forecast_days=None):

    if crop == "مش متأكد" or crop not in CROP_ECONOMICS:
        return None

    econ = CROP_ECONOMICS[crop]
    duration = econ["duration_days"]
    category = econ["category"]
    planting_date = _reference_planting_date(farm.planting_month)
    effective_soil = (soil_profile or {}).get("effective_soil_type") or farm.soil_type
    is_sandy = effective_soil == "رملية"
    is_clay = effective_soil == "طينية"

    daily_m3_note_ar = ""
    daily_m3_note_en = ""
    if live_water is not None:
        daily_m3_note_ar = f" (متوسط المقنن الحي ≈ {live_water['daily_avg_m3_per_feddan']} م³/فدان/يوم)"
        daily_m3_note_en = f" (live avg ≈ {live_water['daily_avg_m3_per_feddan']} m³/feddan/day)"

    soil_irr_note_ar = (
        "يفضّل تقسيم الريات إلى دفعات قصيرة ومتقاربة لتقليل الرشح في التربة الرملية."
        if is_sandy else (
            "تجنّب إغراق التربة الطينية وتأكد من الصرف الجيد بعد كل رية."
            if is_clay else
            "حافظ على رطوبة معتدلة ومتوازنة دون تعطيش أو تغريق."
        )
    ) + daily_m3_note_ar
    soil_irr_note_en = (
        "Split irrigations into shorter, more frequent intervals for sandy soil."
        if is_sandy else (
            "Avoid waterlogging in clay soil and maintain clean drains."
            if is_clay else
            "Maintain even, balanced soil moisture without stress or flooding."
        )
    ) + daily_m3_note_en

    if category == "orchard":
        stage_templates = [
            (
                0.00, 0.15, False,
                ("بداية الموسم والتقليم والخدمة الشتوية/الربيعية", "Season Start, Pruning & Soil Preparation"),
                ("رية تأسيسية معتدلة لتنشيط الجذور.", "Moderate kick-off irrigation to activate roots."),
                ("إضافة السماد العضوي المتحلل والسوبر فوسفات.", "Apply composted organic matter and superphosphate."),
                [
                    ("تقليم الأفرع الجافة والمصابة لفتح قلب الشجرة.", "Prune dry/diseased branches for airflow."),
                    ("تطهير شبكة التنقيط وفحص ضغط النقاطات.", "Flush drip lines and check emitter uniformity."),
                ],
            ),
            (
                0.15, 0.35, True,
                ("التزهير وعقد الثمار (مرحلة حرجة)", "Flowering & Fruit Set (Critical)"),
                ("انتظام الري وتجنب التغريق أو التعطيش لمنع تساقط الأزهار.", "Keep irrigation steady; avoid stress to prevent flower drop."),
                ("دفعات متوازنة من البوتاسيوم والبورون والزنك.", "Balanced potassium, boron, and zinc support."),
                [
                    ("الوقاية من البياض الدقيقي والحشرات الثاقبة/التربس.", "Monitor powdery mildew and thrips."),
                    ("عدم الرش الكيماوي أثناء ذروة نشاط النحل.", "Avoid spraying during peak pollinator activity."),
                ],
            ),
            (
                0.35, 0.70, False,
                ("نمو وتحجيم الثمار", "Fruit Growth & Sizing"),
                ("زيادة مقننات الري تدريجياً مع ارتفاع الحرارة. " + soil_irr_note_ar, "Increase irrigation gradually with rising heat. " + soil_irr_note_en),
                ("تسميد آزوتي وبوتاسي منتظم على دفعات.", "Regular split nitrogen and potassium fertigation."),
                [
                    ("متابعة ذبابة الفاكهة والأكاروسات.", "Scout for fruit flies and mites."),
                    ("إزالة الحشائش تحت الأشجار.", "Control weeds under the canopy."),
                ],
            ),
            (
                0.70, 0.88, False,
                ("التلوين واكتمال النضج", "Coloring & Maturation"),
                ("تقليل الري نسبياً لرفع نسبة السكريات وجودة الثمار.", "Slightly reduce irrigation to boost sugar and quality."),
                ("إيقاف التسميد الآزوتي والتركيز على البوتاسيوم.", "Stop nitrogen; focus on potassium finishing."),
                [
                    ("فحص علامات النضج المحددة للصنف.", "Check cultivar maturity indicators."),
                    ("تجهيز عبوات الجمع المبطنة.", "Prepare padded harvest crates."),
                ],
            ),
            (
                0.88, 1.00, False,
                ("الجمع وما بعد الحصاد", "Harvest & Post-Harvest"),
                ("رية خفيفة بعد الجمع لتعويض إجهاد الأشجار.", "Light post-harvest irrigation to recover tree vigor."),
                ("دفعة تنشيطية خفيفة بعد الحصاد إذا لزم الأمر.", "Light recovery nutrients post-harvest if needed."),
                [
                    ("الجمع في الصباح الباكر وتبريد الثمار في الظل.", "Harvest early morning and keep fruit shaded."),
                    ("فرز الثمار التالفة بعيداً عن الحقل.", "Remove damaged fruit from the orchard."),
                ],
            ),
        ]
    else:
        stage_templates = [
            (
                0.00, 0.08, False,
                ("تجهيز الأرض والزراعة", "Land Preparation & Sowing"),
                ("رية الزراعة (المحاياة الأولى) باعتدال لتسهيل الإنبات.", "Moderate sowing irrigation for uniform germination."),
                ("إضافة السماد العضوي والسوبر فوسفات وسماد البداية قبل الزراعة.", "Incorporate compost, superphosphate, and starter fertilizer."),
                [
                    ("تسوية التربة جيداً (أو عمل مصاطب/خطوط مرتفعة).", "Level soil well or shape raised beds."),
                    ("استخدام تقاوي معتمدة ومعاملة فطرياً.", "Use certified, treated seeds or transplants."),
                ],
            ),
            (
                0.08, 0.25, False,
                ("الإنبات والتأسيس والترقيع", "Germination & Establishment"),
                ("رية المحاياة في الميعاد المناسب. " + soil_irr_note_ar, "Timely establishment irrigation. " + soil_irr_note_en),
                ("الجرعة الأولى من الآزوت (بعد الخف إن وجد).", "First nitrogen split (after thinning if applicable)."),
                [
                    ("ترقيع الجور الغائبة وخف النباتات الزائدة.", "Gap-fill missing hills and thin excess plants."),
                    ("مكافحة الحشائش المبكرة ومراقبة دودة القارض/المن.", "Control early weeds and scout for cutworms/aphids."),
                ],
            ),
            (
                0.25, 0.50, False,
                ("النمو الخضري السريع", "Rapid Vegetative Growth"),
                ("الالتزام بجدول الري المنتظم. " + soil_irr_note_ar, "Follow regular irrigation schedule. " + soil_irr_note_en),
                ("استكمال جرعات النيتروجين وإضافة العناصر الصغرى عند الحاجة.", "Complete nitrogen splits and micronutrients if needed."),
                [
                    ("فحص الأوراق بحثاً عن التبقعات أو الآفات الثاقبة.", "Scout foliage for spots and borers."),
                    ("تطهير المصارف/المساقي أو فحص خطوط التنقيط.", "Keep drains/canals clear or inspect drip lines."),
                ],
            ),
            (
                0.50, 0.72, True,
                ("التزهير والعقد / طرد السنابل (مرحلة حرجة)", "Flowering & Fruit/Grain Set (Critical)"),
                ("ممنوع تعطيش المحصول إطلاقاً في هذه المرحلة الحرجة.", "Avoid any water stress during this critical stage."),
                ("تركيز التسميد على البوتاسيوم والفوسفور والكالسيوم لتثبيت العقد.", "Focus on potassium, phosphorus, and calcium to support set."),
                [
                    ("متابعة تحذيرات الحرارة المرتفعة أو الرطوبة في توقعات الطقس.", "Watch heat and humidity weather alerts closely."),
                    ("تجنب الرش وقت الظهيرة.", "Avoid midday spraying."),
                ],
            ),
            (
                0.72, 0.90, False,
                ("امتلاء المحصول والنضج", "Filling & Maturation"),
                ("تباعد الريات تدريجياً تمهيداً لمرحلة النضج.", "Gradually widen irrigation intervals toward maturity."),
                ("إيقاف التسميد الآزوتي لتحسين جودة التخزين والنضج.", "Stop nitrogen to improve maturity and storage quality."),
                [
                    ("مراقبة الطيور أو أعفان الثمار/الدرنات.", "Monitor for birds or fruit/tuber rots."),
                    ("مراقبة علامات النضج الفسيولوجي.", "Check physiological maturity signs."),
                ],
            ),
            (
                0.90, 1.00, False,
                ("فطام الري والحصاد", "Pre-Harvest Cut-Off & Harvest"),
                ("وقف الري قبل الحصاد بمدة كافية حسب نوع المحصول والتربة.", "Stop irrigation sufficiently ahead of harvest."),
                ("لا يُضاف أي تسميد في هذه المرحلة.", "No fertilizer application during this stage."),
                [
                    ("الحصاد في الأوقات المعتدلة وتجفيف/فرز المحصول جيداً.", "Harvest during mild hours and dry/grade properly."),
                    ("التخزين في مكان جاف وجيد التهوية.", "Store in a dry, well-ventilated facility."),
                ],
            ),
        ]

    stages = []
    for idx, (start_pct, end_pct, critical, title_pair, irr_pair, fert_pair, tasks_pairs) in enumerate(stage_templates, start=1):
        start_day = round(start_pct * duration)
        end_day = max(start_day + 1, round(end_pct * duration))
        start_dt = planting_date + timedelta(days=start_day)
        end_dt = planting_date + timedelta(days=end_day)

        stages.append({
            "stage_number": idx,
            "stage_name": L(lang, *title_pair),
            "start_day": start_day,
            "end_day": end_day,
            "estimated_start_date": start_dt.isoformat(),
            "estimated_end_date": end_dt.isoformat(),
            "estimated_month_name": month_name(lang, start_dt.month),
            "critical": critical,
            "irrigation_guidance": L(lang, *irr_pair),
            "fertilization_guidance": L(lang, *fert_pair),
            "key_actions": [L(lang, a, e) for a, e in tasks_pairs],
        })

    return {
        "crop": crop,
        "crop_en": CROP_PROFILES[crop]["en"],
        "planting_month": farm.planting_month,
        "planting_month_name": month_name(lang, farm.planting_month),
        "reference_planting_date": planting_date.isoformat(),
        "estimated_harvest_date": (planting_date + timedelta(days=duration)).isoformat(),
        "total_duration_days": duration,
        "stages": stages,
    }


# ============================================================
# 21. BETTER CROPS FOR THE FARM
# ============================================================

def rank_alternative_crops(farm, ctx, lang, exclude, top):

    confidence_weight = {"high": 1.0, "medium": 0.95, "low": 0.85}

    ranked = []

    for crop in CROP_PROFILES:

        if crop == exclude:
            continue

        evaluation = evaluate_crop(crop, farm, ctx, lang)

        if evaluation["suitability"] == "not_suitable":
            continue

        if any(
            c["factor"] == "region" and c["status"] != "good"
            for c in evaluation["checks"]
        ):
            continue

        ranked.append(evaluation)

    ranked.sort(
        key=lambda item: -(item["score"] * confidence_weight[item["confidence"]])
    )

    return ranked[:top]


# ============================================================
# 21B. SIDE-BY-SIDE TWO-CROP COMPARISON
# ============================================================

def _crop_comparison_card(
    crop, evaluation, farm, lang,
    weather_summary=None, season=None, forecast_days=None,
    soil_profile=None, elevation_data=None, market_indicators=None
):

    profile = CROP_PROFILES[crop]
    live_water = (
        calculate_live_crop_water_demand(
            crop, farm, weather_summary, season, forecast_days, soil_profile, elevation_data, lang
        )
        if weather_summary is not None else None
    )
    econ_water = calculate_economics_and_water(
        crop, farm, lang,
        live_water=live_water,
        market_indicators=market_indicators,
        apply_manual_overrides=False,
    )
    timing_alert = build_planting_timing_alert(crop, farm.planting_month, lang)
    water_demand_labels = {
        "low": ("منخفض", "Low"),
        "medium": ("متوسط", "Medium"),
        "high": ("مرتفع", "High"),
        "very_high": ("مرتفع جداً", "Very High"),
    }
    wd_ar, wd_en = water_demand_labels.get(profile["water"], ("متوسط", "Medium"))

    return {
        "crop": crop,
        "crop_en": profile["en"],
        "suitability": evaluation["suitability"],
        "suitability_label": suitability_label(lang, evaluation["suitability"]),
        "suitability_score": evaluation["score"],
        "suitability_confidence": evaluation["confidence"],
        "water_demand_level": L(lang, wd_ar, wd_en),
        "recommended_irrigation": econ_water["water_savings"]["recommended_method"] if econ_water else None,
        "water_m3_per_feddan": econ_water["water_savings"]["recommended_method_m3_per_feddan"] if econ_water else None,
        "water_saved_vs_flood_m3": econ_water["water_savings"]["saved_vs_flood_m3_per_feddan"] if econ_water else None,
        "net_profit_per_feddan_egp": econ_water["economics"]["net_profit_per_feddan_egp"] if econ_water else None,
        "roi_percent": econ_water["economics"]["roi_percent"] if econ_water else None,
        "risks_count": len(profile["risks"]),
        "risks": [L(lang, a, e) for a, e in profile["risks"]],
        "planting_timing_status": timing_alert["status"] if timing_alert else None,
        "checks": [
            {"factor": c["factor"], "status": c["status"], "message": c["message"]}
            for c in evaluation["checks"]
        ],
    }


def compare_two_crops(
    primary_crop, secondary_crop, farm, ctx, lang,
    forecast_days=None, market_indicators=None
):

    eval_1 = evaluate_crop(primary_crop, farm, ctx, lang)
    eval_2 = evaluate_crop(secondary_crop, farm, ctx, lang)

    weather_summary = ctx.get("weather_summary")
    season = ctx.get("season")
    soil_profile = ctx.get("soil_profile")
    elevation_data = ctx.get("elevation_data")

    card_1 = _crop_comparison_card(
        primary_crop, eval_1, farm, lang,
        weather_summary=weather_summary, season=season, forecast_days=forecast_days,
        soil_profile=soil_profile, elevation_data=elevation_data, market_indicators=market_indicators
    )
    card_2 = _crop_comparison_card(
        secondary_crop, eval_2, farm, lang,
        weather_summary=weather_summary, season=season, forecast_days=forecast_days,
        soil_profile=soil_profile, elevation_data=elevation_data, market_indicators=market_indicators
    )

    score_1 = card_1["suitability_score"]
    score_2 = card_2["suitability_score"]
    water_1 = card_1["water_m3_per_feddan"] or 0
    water_2 = card_2["water_m3_per_feddan"] or 0

    if score_1 > score_2:
        winner = "primary"
        winner_card = card_1
    elif score_2 > score_1:
        winner = "secondary"
        winner_card = card_2
    else:
        if water_1 < water_2:
            winner = "primary"
            winner_card = card_1
        elif water_2 < water_1:
            winner = "secondary"
            winner_card = card_2
        else:
            winner = "tie"
            winner_card = None

    score_diff = abs(score_1 - score_2)
    water_diff = abs(water_1 - water_2)

    if winner == "tie":
        summary = L(
            lang,
            f"تعادل المحصولان ({card_1['crop']} و{card_2['crop']}) في درجة الملاءمة ({score_1}/100) والاستهلاك المائي؛ يمكنك المفاضلة حسب العائد المتوقع وسهولة التسويق.",
            f"Both crops ({card_1['crop_en']} and {card_2['crop_en']}) tied in suitability ({score_1}/100) and water use; choose based on expected profit and local market demand."
        )
    else:
        loser_card = card_2 if winner == "primary" else card_1
        winner_name = winner_card["crop"] if lang == "ar" else winner_card["crop_en"]
        loser_name = loser_card["crop"] if lang == "ar" else loser_card["crop_en"]
        if score_diff > 0:
            summary = L(
                lang,
                f"الأفضل لظروف أرضك حالياً هو «{winner_name}» بدرجة ملاءمة {winner_card['suitability_score']}/100 مقابل {loser_card['suitability_score']}/100 لمحصول «{loser_name}» (فارق {score_diff} نقطة).",
                f"The better fit for your conditions is {winner_name} with a score of {winner_card['suitability_score']}/100 vs {loser_card['suitability_score']}/100 for {loser_name} (+{score_diff} pts)."
            )
        else:
            summary = L(
                lang,
                f"تساوى المحصولان في درجة الملاءمة ({score_1}/100)، لكن «{winner_name}» يتفوق باستهلاك مياه أقل بنحو {water_diff:,} م³/فدان.",
                f"Both crops scored {score_1}/100, but {winner_name} wins by saving ~{water_diff:,} m³/feddan in water use."
            )

    return {
        "primary": card_1,
        "secondary": card_2,
        "winner": winner,
        "winner_crop": winner_card["crop"] if winner_card else None,
        "winner_crop_en": winner_card["crop_en"] if winner_card else None,
        "score_difference": score_diff,
        "water_difference_m3_per_feddan": water_diff,
        "summary": summary,
    }


# ============================================================
# 21C. TRANSPARENT METHODOLOGY ("HOW WE CALCULATE")
# ============================================================

def build_methodology(lang, evaluation=None):

    methodology = {
        "title": L(
            lang,
            "منهجية الحساب والتقييم في N.A.D.A",
            "N.A.D.A Evaluation & Calculation Methodology"
        ),
        "data_sources": [
            {
                "name": "NASA POWER Live Daily & SMAP Soil Moisture (AG Community)",
                "purpose": L(
                    lang,
                    "بيانات الطقس الفعلية ورطوبة التربة السطحية (GWETTOP) ورطوبة الجذور (GWETROOT من SMAP) والبخر-نتح الفعلي (EVPTRNS) والإشعاع الشمسي (ALLSKY_SFC_SW_DWN).",
                    "Observed daily weather plus surface wetness (GWETTOP), SMAP root-zone wetness (GWETROOT), evapotranspiration flux (EVPTRNS), and solar radiation."
                ),
            },
            {
                "name": "NASA SRTM Elevation (via Open-Elevation API)",
                "purpose": L(
                    lang,
                    "جلب ارتفاع الأرض عن سطح البحر لتقييم كفاءة الري بالغمر مقابل التنقيط ومخاطر ارتفاع الماء الأرضي أو تكلفة الضخ.",
                    "Digital elevation above sea level to assess flood vs drip irrigation efficiency, water-table risk, and pumping head."
                ),
            },
            {
                "name": "ISRIC SoilGrids REST API v2.0 (Hybrid Soil Engine)",
                "purpose": L(
                    lang,
                    "سحب نسبة الطين والرمل ودرجة الحموضة pH تلقائياً، مع إعطاء الأولوية القصوى لنتائج التحليل المعملي اليدوي إن أرسلها المزارع.",
                    "Automated clay, sand, and pH retrieval by coordinates, overridden with top priority by manual laboratory results when supplied."
                ),
            },
            {
                "name": "Open-Meteo 16-Day Forecast + FAO ET0 & Seasonal Outlook",
                "purpose": L(
                    lang,
                    "توقعات الطقس ومعادلة البخر-نتح للفاو (et0_fao_evapotranspiration) لبناء جدول الري والتسميد المستقبلي لـ 16 يوماً.",
                    "16-day weather forecast and FAO Penman-Monteith ET0 for operational irrigation and fertilization scheduling."
                ),
            },
            {
                "name": "Live FX & Commodity/Fertilizer Cost Indexing (open.er-api.com)",
                "purpose": L(
                    lang,
                    "تحديث التكلفة التقديرية وأسعار المحاصيل ديناميكياً وفق سعر الصرف اللحظي للجنيه ومؤشر الأسمدة مع مرونة التعديل اليدوي.",
                    "Dynamic cost and crop-price adjustment using live USD/EGP exchange rates and input indices with manual overrides."
                ),
            },
        ],
        "scoring_model": {
            "equation": "Score = (Sum(Status_Value * Factor_Weight) / Sum(Evaluated_Weights)) * 100",
            "status_values": {
                "good": 1.0,
                "warning": 0.5,
                "bad": 0.0,
                "info": L(lang, "مستبعد من المقام حتى لا يؤثر سلباً أو إيجاباً عند نقص البيانات", "Excluded from denominator when data is unknown"),
            },
            "factor_weights": {
                "planting_month": 3.0,
                "soil": 3.0,
                "temperature": 2.0,
                "drainage": 2.0,
                "region": "1.0 - 2.0 (crop-dependent)",
                "irrigation": 1.5,
                "water_source": 1.0,
                "salinity": 1.0,
                "humidity": 1.0,
                "regional_heat": 1.0,
                "soil_ph": 0.5,
                "elevation_irrigation": 0.5,
                "irrigation_soil": 0.5,
                "rain": 0.5,
                "wind": 0.5,
            },
            "classification_rules": {
                "not_suitable": L(
                    lang,
                    "إذا حصل أحد العوامل الأساسية (شهر الزراعة، الحرارة، التربة) على تقييم (bad).",
                    "Triggered when any core factor (planting_month, temperature, soil) is rated 'bad'."
                ),
                "suitable_with_conditions": L(
                    lang,
                    "لا توجد عوامل أساسية مرفوضة، لكن يوجد تحذير (warning) أو عامل ثانوي غير مثالي، أو المحصول خارج محافظاته المعتادة (حد أقصى للدرجة 60 أو 75).",
                    "No core factor is 'bad', but at least one warning/bad secondary check exists, or regional cap (60/75) applies."
                ),
                "suitable": L(
                    lang,
                    "جميع العوامل المقيمة حصلت على (good) بدون تحذيرات.",
                    "All evaluated checks are 'good' with no warnings."
                ),
            },
            "confidence_rules": {
                "formula": "Judged_Core_Weight / Total_Core_Weight (11.5)",
                "thresholds": {
                    "high": ">= 85%",
                    "medium": "65% - 84%",
                    "low": "< 65%",
                },
            },
        },
        "economics_and_water_formulas": {
            "live_water_demand": "Net_M3_Feddan = ((Blended_ET0 * Kc * Season_Days) - Effective_Rain - SMAP_Moisture_Credit) * 4.20083 / Irrigation_Efficiency",
            "gross_revenue": "Expected_Yield_Ton_Per_Feddan * Dynamic_Price_Per_Ton_EGP",
            "net_profit": "Gross_Revenue - Dynamic_Cost_Per_Feddan_EGP",
            "roi_percent": "(Net_Profit / Dynamic_Cost_Per_Feddan_EGP) * 100",
            "water_savings_vs_flood": {
                "drip_efficiency": "90% vs Flood 58% (~35% water saved)",
                "sprinkler_efficiency": "78% vs Flood 58% (~25% water saved)",
                "rice_awd": "15% saving via Alternate Wetting & Drying (AWD)",
                "area_conversion": "1 Hectare = 2.38095 Feddan",
            },
        },
    }

    if evaluation is not None:
        methodology["current_run_breakdown"] = {
            "crop": evaluation["crop"],
            "weighted_points": evaluation["weighted_points"],
            "evaluated_weight_sum": evaluation["scored_weight_sum"],
            "final_score": evaluation["score"],
            "confidence": evaluation["confidence"],
            "factors": [
                {
                    "factor": c["factor"],
                    "status": c["status"],
                    "weight": c["weight"],
                    "points_earned": (
                        round(STATUS_VALUE[c["status"]] * c["weight"], 2)
                        if c["status"] in STATUS_VALUE else None
                    ),
                }
                for c in evaluation["checks"]
            ],
        }

    return methodology


# ============================================================
# 22. CREATE INITIAL FARM ASSESSMENT
# ============================================================

def create_initial_assessment(
    farm, weather_summary, period_end, lang,
    climatology=None, season=None, soil_profile=None, elevation_data=None
):

    observations = []

    missing_information = []

    risks = []

    recommendations = []

    checks_output = []

    alternative_crops = []

    ctx = build_climate_context(
        weather_summary, period_end, climatology, farm.planting_month,
        season=season, soil_profile=soil_profile, elevation_data=elevation_data
    )

    confidence_code = None

    unjudged_keys = []

    irrigation_text = None

    timing_alert = None

    primary_evaluation = None

    if farm.standing_water == "نعم":

        observations.append(L(
            lang,
            "المزارع أبلغ عن وجود مياه راكدة بعد الري. يجب تقييم صرف الحقل.",
            "The farmer reports standing water after irrigation. "
            "Field drainage should be assessed."
        ))

    elif farm.standing_water == "أحيانًا":

        observations.append(L(
            lang,
            "المزارع يلاحظ أحيانًا وجود مياه راكدة. يحتاج الأمر لفحص مدى تكرارها ومدتها.",
            "The farmer sometimes observes standing water. "
            "Its frequency and duration need investigation."
        ))

    if soil_profile:
        observations.append(L(
            lang,
            f"ℹ مصدر بيانات التربة المعتمد: {soil_profile['source_label']} "
            f"(طين {soil_profile.get('clay_percent')}% • رمل {soil_profile.get('sand_percent')}% • pH {soil_profile.get('ph')}).",
            f"ℹ Active soil data source: {soil_profile['source_label']} "
            f"(Clay {soil_profile.get('clay_percent')}% • Sand {soil_profile.get('sand_percent')}% • pH {soil_profile.get('ph')})."
        ))

    if elevation_data:
        observations.append(L(
            lang,
            f"ℹ ارتفاع الأرض (NASA SRTM): {elevation_data['elevation_m']} م فوق سطح البحر — {elevation_data['flood_vs_drip_note_ar']}",
            f"ℹ Field elevation (NASA SRTM): {elevation_data['elevation_m']} m ASL — {elevation_data['flood_vs_drip_note_en']}"
        ))

    if weather_summary.get("average_root_zone_soil_wetness") is not None:
        gwet_pct = round(weather_summary["average_root_zone_soil_wetness"] * 100, 1)
        observations.append(L(
            lang,
            f"ℹ رطوبة منطقة الجذور من قمر ناسا (SMAP GWETROOT): {gwet_pct}% خلال آخر 30 يوماً.",
            f"ℹ NASA SMAP root-zone soil wetness (GWETROOT): {gwet_pct}% over the last 30 days."
        ))

    if farm.soil_type == "لا أعرف" and not (soil_profile and soil_profile.get("inferred_texture_from_data")):

        missing_information.append(L(
            lang,
            "تحديد قوام التربة",
            "Soil texture identification"
        ))

    if farm.soil_lab_results is None:

        missing_information.append(L(
            lang,
            "تحليل معملي للتربة: الحموضة والملوحة والعناصر الغذائية (أو الاعتماد على مسح SoilGrids)",
            "Laboratory soil analysis: pH, salinity and nutrients (SoilGrids satellite estimate used)"
        ))

    if farm.irrigation_water_results is None:

        missing_information.append(L(
            lang,
            "تحليل جودة مياه الري",
            "Irrigation water quality analysis"
        ))

    if farm.soil_photo_url is None:

        missing_information.append(L(
            lang,
            "صورة التربة (اختيارية)",
            "Optional soil photograph"
        ))

    if farm.crop_photo_url is None:

        missing_information.append(L(
            lang,
            "صورة المحصول (اختيارية)",
            "Optional crop photograph"
        ))

    if farm.farm_area is None:

        missing_information.append(L(
            lang,
            "مساحة المزرعة",
            "Farm area"
        ))

    # ------------------------------------------------------------
    # Crop selected by the farmer
    # ------------------------------------------------------------

    if farm.crop != "مش متأكد":

        evaluation = evaluate_crop(farm.crop, farm, ctx, lang)
        primary_evaluation = evaluation

        profile = CROP_PROFILES[farm.crop]

        name = evaluation["name"]

        suitability_text = suitability_label(lang, evaluation["suitability"])

        status_text = L(
            lang,
            f"{suitability_text} — درجة الملاءمة {evaluation['score']}/100 (تقييم أولي إرشادي)",
            f"{suitability_text} — suitability score {evaluation['score']}/100 (initial guidance)"
        )

        observations.append(L(
            lang,
            f"تقييم {name}: {suitability_text} ({evaluation['score']}/100).",
            f"Evaluation of {name}: {suitability_text} ({evaluation['score']}/100)."
        ))

        timing_alert = build_planting_timing_alert(farm.crop, farm.planting_month, lang)
        if timing_alert and timing_alert["status"] != "on_time":
            observations.append(timing_alert["message"])

        confidence_code = evaluation["confidence"]

        unjudged_keys = [
            f for f in evaluation["unjudged"] if f in FACTOR_NAMES
        ]

        if confidence_code != "high" and unjudged_keys:

            separator = "، " if lang == "ar" else ", "

            names = separator.join(
                L(lang, *FACTOR_NAMES[f]) for f in unjudged_keys
            )

            status_text += L(
                lang,
                f" — درجة أولية: لم تُقيَّم ({names})",
                f" — provisional score: not judged ({names})"
            )

            observations.append(L(
                lang,
                f"ℹ الدرجة أولية لأن هذه العوامل لم تُقيَّم: {names}.",
                f"ℹ The score is provisional because these factors were not judged: {names}."
            ))

        season_line = season_text(lang, ctx["season"])

        if season_line:

            observations.append(season_line)

        icons = {"bad": "✖ ", "warning": "⚠ "}

        for status in ("bad", "warning"):

            for check in evaluation["checks"]:

                if check["status"] == status:
                    observations.append(icons[status] + check["message"])

        for check in evaluation["checks"]:

            if check["status"] == "good":
                observations.append("✔ " + check["message"])

        for check in evaluation["checks"]:

            if check["status"] == "info":
                observations.append("ℹ " + check["message"])

        checks_output = [
            {
                "factor": c["factor"],
                "status": c["status"],
                "message": c["message"],
            }
            for c in evaluation["checks"]
        ]

        irrigation_text = irrigation_recommendation(
            farm.crop, farm, lang, soil_profile=soil_profile, elevation_data=elevation_data
        )

        observations.append(irrigation_text)

        for arabic, english in profile["risks"]:

            text = L(lang, arabic, english)

            risks.append(text)

            observations.append(L(lang, "خطر محتمل: ", "Possible risk: ") + text)

        advice = [L(lang, a, e) for a, e in profile["tips"]]

        effective_soil = (soil_profile or {}).get("effective_soil_type") or farm.soil_type
        if effective_soil in SOIL_TIPS:

            advice.append(L(lang, *SOIL_TIPS[effective_soil]))

        for arabic, english in PROBLEM_ADVICE[farm.agricultural_problem]:

            advice.append(L(lang, arabic, english))

        for text in advice:

            recommendations.append(text)

            observations.append(L(lang, "نصيحة: ", "Tip: ") + text)

        if evaluation["suitability"] != "suitable":

            better = [
                item
                for item in rank_alternative_crops(
                    farm, ctx, lang, farm.crop, 3
                )
                if item["score"] > evaluation["score"]
            ]

            if better:

                separator = "، " if lang == "ar" else ", "

                summary = separator.join(
                    f"{item['name']} ({item['score']}/100)"
                    for item in better
                )

                observations.append(L(
                    lang,
                    f"محاصيل أنسب لظروفك: {summary}.",
                    f"Crops that fit your conditions better: {summary}."
                ))

                alternative_crops = [
                    {
                        "crop": item["crop"],
                        "crop_en": CROP_PROFILES[item["crop"]]["en"],
                        "score": item["score"],
                        "suitability": item["suitability"],
                    }
                    for item in better
                ]

        suitability_code = evaluation["suitability"]

        score = evaluation["score"]

    # ------------------------------------------------------------
    # Crop not selected: suggest the best fitting crops
    # ------------------------------------------------------------

    else:

        ranked = rank_alternative_crops(farm, ctx, lang, None, 5)

        suitability_code = "not_evaluated"

        score = None

        separator = "، " if lang == "ar" else ", "

        if ranked:

            summary = separator.join(
                f"{item['name']} ({item['score']}/100)" for item in ranked
            )

            status_text = L(
                lang,
                f"لم تختر محصولًا. أنسب المحاصيل لظروفك: {summary}",
                f"No crop selected. Best fitting crops for your conditions: {summary}"
            )

            observations.append(L(
                lang,
                "لم يتم اختيار محصول. هذه أنسب المحاصيل لظروفك حاليًا:",
                "No crop has been selected. These crops fit your conditions best:"
            ))

            for item in ranked:

                suitability_text = suitability_label(lang, item["suitability"])

                observations.append(
                    f"• {item['name']}: {item['score']}/100 — {suitability_text}"
                )

            alternative_crops = [
                {
                    "crop": item["crop"],
                    "crop_en": CROP_PROFILES[item["crop"]]["en"],
                    "score": item["score"],
                    "suitability": item["suitability"],
                }
                for item in ranked
            ]

        else:

            status_text = L(
                lang,
                "لم تختر محصولًا، ولم نجد محصولًا مناسبًا بين المحاصيل المتاحة لهذا الشهر والتربة.",
                "No crop selected, and no crop in the table fits this month and soil."
            )

            observations.append(status_text)

        effective_soil = (soil_profile or {}).get("effective_soil_type") or farm.soil_type
        if effective_soil in SOIL_TIPS:

            soil_tip = L(lang, *SOIL_TIPS[effective_soil])

            recommendations.append(soil_tip)

            observations.append(L(lang, "نصيحة: ", "Tip: ") + soil_tip)

        for arabic, english in PROBLEM_ADVICE[farm.agricultural_problem]:

            text = L(lang, arabic, english)

            recommendations.append(text)

            observations.append(L(lang, "نصيحة: ", "Tip: ") + text)

        season_line = season_text(lang, ctx["season"])

        if season_line:

            observations.append(season_line)

    # ------------------------------------------------------------
    # Regional notes
    # ------------------------------------------------------------

    if farm.governorate in COASTAL_DELTA_GOVERNORATES:

        observations.append(L(
            lang,
            "ℹ المناطق الساحلية بشمال الدلتا معرضة لارتفاع الماء الأرضي والملوحة؛ تابع الصرف وحلّل الملوحة.",
            "ℹ Coastal northern Delta areas are prone to high water tables and salinity; watch drainage and test salinity."
        ))

    if farm.governorate in DESERT_GOVERNORATES:

        observations.append(L(
            lang,
            "ℹ الأراضي الصحراوية غالبًا رملية وتعتمد على مياه الآبار؛ حلّل ملوحة المياه وفضّل الري بالتنقيط.",
            "ℹ Desert lands are often sandy and rely on well water; test water salinity and prefer drip irrigation."
        ))

    return {

        "observations": observations,

        "missing_information": missing_information,

        "crop_suitability_status": status_text,

        "assessment_type": L(lang, "تقييم أولي إرشادي", "Initial assessment"),

        "suitability": suitability_code,

        "suitability_score": score,

        "suitability_confidence": confidence_code,

        "unjudged_factors": unjudged_keys,

        "planting_timing_alert": timing_alert,

        "planting_month_climate": (
            {
                k: ctx["season"][k]
                for k in ("source", "temp", "rain_total", "humidity", "wind")
            }
            if ctx["season"] else None
        ),

        "checks": checks_output,

        "irrigation_recommendation": irrigation_text,

        "risks": risks,

        "recommendations": recommendations,

        "alternative_crops": alternative_crops,

        "language": lang,

    }, primary_evaluation, ctx


# ============================================================
# 23. BUILD LIMITATIONS
# ============================================================

def build_limitations(location, weather_summary, farm, lang, season=None, forecast=None, soil_profile=None):

    limitations = []

    if location["coordinate_type"] != "GPS":

        limitations.append(L(
            lang,
            "إحداثيات المحافظة تقريبية وقد لا تمثل موقع المزرعة بدقة.",
            "Governorate coordinates are approximate "
            "and may not represent the exact farm location."
        ))

    limitations.append(L(
        lang,
        "بيانات ناسا باور ورطوبة SMAP بيانات بيئية على شبكة مربعات وليست قياسات مباشرة من حقل المزارع.",
        "NASA POWER and SMAP provide gridded environmental data, "
        "not direct point measurements from the farmer's field."
    ))

    if weather_summary.get("data_quality", {}).get("nasa_live_status") == "fallback":

        limitations.append(L(
            lang,
            "تم استخدام متوسطات مناخية احتياطية لمصر مؤقتاً بسبب بطء استجابة خادم ناسا عن المهلة المحددة.",
            "Regional Egypt climate fallback was used temporarily due to NASA POWER response latency."
        ))

    limitations.append(L(
        lang,
        "ملخص الطقس يصف بيانات يومية تاريخية لآخر 30 يومًا وليس توقعًا للطقس.",
        "The weather summary describes historical daily data, "
        "not a weather forecast."
    ))

    if season is not None and season["source"] == "climatology":

        limitations.append(L(
            lang,
            "مناخ شهر الزراعة هو متوسط عدة سنوات من ناسا، ولا يعني طقس هذا الموسم تحديدًا.",
            "The planting-month climate is a multi-year NASA average and does not "
            "describe this particular season's weather."
        ))

    if forecast is not None:

        limitations.append(L(
            lang,
            "توقعات الطقس والبخر-نتح (FAO ET0) من Open-Meteo تقريبية وتتغير يوميًا، وتوقع الشهر كله اتجاه عام استرشادي.",
            "Weather forecasts and FAO ET0 from Open-Meteo are approximate and change daily; "
            "the 30-day outlook is a general tendency."
        ))

    if soil_profile and soil_profile.get("source_code") in ("soilgrids_satellite", "regional_fallback"):

        limitations.append(L(
            lang,
            "بيانات التربة (SoilGrids / التقدير الإقليمي) تقديرية بدقة 250م؛ إدخال تحليل معملي يدوي يرفع دقة القرار.",
            "SoilGrids/regional soil metrics are 250m gridded estimates; providing a manual lab soil test yields top precision."
        ))

    limitations.append(L(
        lang,
        "تقييم ملاءمة المحصول إرشادي مبني على قواعد ومتطلبات تقريبية للمحاصيل، "
        "وليس تقييمًا علميًا معمليًا ولا يغني عن المهندس الزراعي.",
        "Crop suitability is an initial rule-based guidance built on approximate "
        "crop requirements. It is not a scientific or laboratory evaluation and "
        "does not replace an agricultural engineer."
    ))

    limitations.append(L(
        lang,
        "الأرقام الاقتصادية وتوفير المياه تقديرات ديناميكية مرتبطة بسعر الصرف ومؤشرات السوق ومتوسطات المقننات المائية، وتسمح بالتعديل اليدوي.",
        "Economic and water-saving figures are dynamic estimates tied to FX rates and market indices, with manual override support."
    ))

    limitations.append(L(
        lang,
        "الصور الاختيارية تُسجَّل كمراجع فقط ولا تُستخدم تلقائيًا في التشخيص العلمي.",
        "Optional photographs are recorded as references. "
        "They are not automatically used for scientific diagnosis."
    ))

    if farm.soil_lab_results is not None:

        limitations.append(L(
            lang,
            "تم إعطاء الأولوية القصوى لنتائج التحليل المعملي للتربة المدخلة يدوياً في التقييم الهجين.",
            "Manually entered soil laboratory results were prioritized in the hybrid soil assessment."
        ))

    return limitations


# ============================================================
# 23B. FORECAST SECTION (ALERTS AND OUTLOOK)
# ============================================================

def _day_list(lang, days, limit=3):

    parts = [f"{d['date'][8:10]}/{d['date'][5:7]}" for d in days[:limit]]

    text = ("، " if lang == "ar" else ", ").join(parts)

    if len(days) > limit:

        text += L(lang, " وغيرها", " and others")

    return text


def build_forecast_alerts(farm, days, lang):

    alerts = []

    crop = farm.crop

    selected = crop != "مش متأكد"

    profile = CROP_PROFILES[crop] if selected else None

    name = crop_name(lang, crop) if selected else ""

    cn = f"محصول {name}"

    heat_limit = profile["temp_abs"][1] + 4 if selected else 40

    hot = [d for d in days if d["t_max"] >= heat_limit]

    if hot:

        peak = round(max(d["t_max"] for d in hot), 1)

        when = _day_list(lang, hot)

        if selected:

            alerts.append(L(
                lang,
                f"⚠ ارتفاع شديد في الحرارة متوقع (حتى {peak}°م) أيام {when}؛ {cn} يتأثر بالحرارة العالية. "
                "اروِ في الصباح الباكر أو المساء وأجّل التسميد.",
                f"⚠ Very high heat expected (up to {peak}°C) on {when}; {name} is affected by high heat. "
                "Irrigate early morning or evening and postpone fertilizing."
            ))

        else:

            alerts.append(L(
                lang,
                f"⚠ موجة حر متوقعة (حتى {peak}°م) أيام {when}؛ تجنب الزراعة والتسميد في هذه الأيام.",
                f"⚠ Heat wave expected (up to {peak}°C) on {when}; avoid planting and fertilizing on these days."
            ))

    frost = [d for d in days if d["t_min"] <= 2]

    low = profile["temp_abs"][0] if selected else None

    if frost and (not selected or low >= 5):

        lowest = round(min(d["t_min"] for d in frost), 1)

        when = _day_list(lang, frost)

        who = L(lang, f"{cn} حساس للبرد", f"{name} is cold-sensitive") if selected else L(
            lang, "المحاصيل الحساسة للبرد معرضة للضرر", "cold-sensitive crops can be damaged"
        )

        alerts.append(L(
            lang,
            f"⚠ خطر صقيع (حتى {lowest}°م) أيام {when}؛ {who}. غطِّ الشتلات الصغيرة واروِ قبل الليل.",
            f"⚠ Frost risk (down to {lowest}°C) on {when}; {who}. Cover young plants and irrigate before nightfall."
        ))

    elif selected:

        cold = [d for d in days if d["t_min"] <= low - 4]

        if cold:

            lowest = round(min(d["t_min"] for d in cold), 1)

            alerts.append(L(
                lang,
                f"⚠ برد متوقع (حتى {lowest}°م) أيام {_day_list(lang, cold)}؛ {cn} يتأثر بالبرد، فاحمِ الشتلات.",
                f"⚠ Cold expected (down to {lowest}°C) on {_day_list(lang, cold)}; {name} is affected by cold, protect the plants."
            ))

    rainy = [d for d in days if d["rain_mm"] >= 5]

    if rainy:

        peak = round(max(d["rain_mm"] for d in rainy), 1)

        extra = ""

        if selected and profile["waterlog"] in ("low", "very_low"):

            extra = L(
                lang,
                f"، و{cn} لا يتحمل تشبع التربة، فتأكد من نظافة المصارف",
                f", and {name} does not tolerate waterlogging, so keep the drains clean"
            )

        alerts.append(L(
            lang,
            f"☔ أمطار متوقعة أيام {_day_list(lang, rainy)} (حتى {peak} مم)؛ أجّل الري في هذه الأيام وتجنب رش المبيدات{extra}.",
            f"☔ Rain expected on {_day_list(lang, rainy)} (up to {peak} mm); skip irrigation on these days and avoid spraying{extra}."
        ))

    windy = [
        d for d in days
        if d["wind_max_kmh"] is not None and d["wind_max_kmh"] >= 40
    ]

    if windy:

        peak = round(max(d["wind_max_kmh"] for d in windy))

        extra = ""

        if selected and profile["wind_sensitive"]:

            extra = L(
                lang,
                f"، و{cn} حساس للرياح، فتأكد من مصدات الرياح والدعامات",
                f", and {name} is wind-sensitive, so check windbreaks and supports"
            )

        alerts.append(L(
            lang,
            f"💨 رياح قوية متوقعة أيام {_day_list(lang, windy)} (حتى {peak} كم/س)؛ لا ترش مبيدات في هذه الأيام{extra}.",
            f"💨 Strong wind expected on {_day_list(lang, windy)} (up to {peak} km/h); do not spray on these days{extra}."
        ))

    total_rain = sum(d["rain_mm"] for d in days)

    average_max = sum(d["t_max"] for d in days) / len(days)

    if (
        selected
        and profile["water"] in ("high", "very_high")
        and total_rain < 1
        and average_max >= 30
    ):

        alerts.append(L(
            lang,
            f"💧 الجو جاف وحار ولا أمطار متوقعة في الأسبوعين القادمين، و{cn} يحتاج مياهًا كثيرة؛ حافظ على الري المنتظم.",
            f"💧 Dry and warm with no rain expected, and {name} needs a lot of water; keep irrigation regular."
        ))

    if not alerts:

        alerts.append(L(
            lang,
            "✔ لا توجد تحذيرات جوية مهمة في الأسبوعين القادمين." if not selected
            else f"✔ لا توجد تحذيرات جوية مهمة {arabic_lam(name)} في الأسبوعين القادمين.",
            "✔ No important weather alerts for the next two weeks."
        ))

    return alerts


def summarize_outlook(outlook, climatology, lang):

    dates = sorted(outlook)[:OUTLOOK_DAYS]

    if len(dates) < OUTLOOK_MIN_DAYS:

        return None

    count = len(dates)

    temperature = round(sum(outlook[d]["temp"] for d in dates) / count, 1)

    rain = round(sum(outlook[d]["rain"] for d in dates), 1)

    normal_temp = None

    normal_rain = None

    if climatology:

        temps = []

        rains = []

        for day in dates:

            month = climatology.get(int(day[5:7]))

            if not month:
                continue

            if month["temp"] is not None:
                temps.append(month["temp"])

            if month["rain_mm_day"] is not None:
                rains.append(month["rain_mm_day"])

        if len(temps) == count:

            normal_temp = round(sum(temps) / count, 1)

        if len(rains) == count:

            normal_rain = round(sum(rains), 1)

    ar = f"الاتجاه العام للـ{count} يومًا القادمة: متوسط الحرارة نحو {temperature}°م"

    en = f"General tendency for the next {count} days: mean temperature about {temperature}°C"

    if normal_temp is not None:

        difference = round(temperature - normal_temp, 1)

        if difference >= 1:
            ar += f" (أعلى من المعتاد بنحو {difference}°م)"
            en += f" ({difference}°C above normal)"
        elif difference <= -1:
            ar += f" (أقل من المعتاد بنحو {abs(difference)}°م)"
            en += f" ({abs(difference)}°C below normal)"
        else:
            ar += " (قريب من المعتاد)"
            en += " (close to normal)"

    ar += f"، والأمطار نحو {rain} مم"

    en += f", rain about {rain} mm"

    if normal_rain is not None:

        if rain - normal_rain >= 5:
            ar += " (أكثر من المعتاد)"
            en += " (more than normal)"
        elif normal_rain - rain >= 5:
            ar += " (أقل من المعتاد)"
            en += " (less than normal)"

    ar += ". هذا اتجاه عام ضعيف الدقة وليس توقعًا لكل يوم."

    en += ". This is a low-accuracy general tendency, not a day-by-day forecast."

    return {
        "period_days": count,
        "average_temperature_c": temperature,
        "total_rainfall_mm": rain,
        "normal_temperature_c": normal_temp,
        "normal_rainfall_mm": normal_rain,
        "text": L(lang, ar, en),
    }


def build_forecast_section(farm, forecast_payload, outlook, climatology, lang):

    days = (forecast_payload or {}).get("days")
    if not days and not outlook:
        return None

    return {

        "source": "Open-Meteo (with FAO ET0)" if not (forecast_payload or {}).get("is_fallback") else "Open-Meteo Fallback",

        "days": [
            {
                "date": d["date"],
                "temperature_max_c": round(d["t_max"], 1),
                "temperature_min_c": round(d["t_min"], 1),
                "rain_mm": round(d["rain_mm"], 1),
                "wind_max_kmh": (
                    round(d["wind_max_kmh"])
                    if d["wind_max_kmh"] is not None else None
                ),
                "et0_mm": round(d["et0_mm"], 2) if d.get("et0_mm") is not None else None,
            }
            for d in (days or [])
        ],

        "alerts": build_forecast_alerts(farm, days, lang) if days else [],

        "outlook": (
            summarize_outlook(outlook, climatology, lang) if outlook else None
        ),

    }


# ============================================================
# 24. PARALLEL ENVIRONMENTAL FETCH & MAIN ANALYSIS ENDPOINTS
# ============================================================

def _fetch_farm_environmental_data(latitude, longitude, governorate=None, soil_type=None):

    with ThreadPoolExecutor(max_workers=6) as pool:

        nasa_daily_future = pool.submit(get_nasa_data, latitude, longitude)
        climatology_future = pool.submit(_safe_call, get_nasa_climatology, latitude, longitude)
        forecast_future = pool.submit(_safe_call, get_short_forecast, latitude, longitude)
        outlook_future = pool.submit(_safe_call, get_outlook, latitude, longitude)
        soilgrids_future = pool.submit(
            _safe_call, get_soilgrids_data, latitude, longitude, governorate, soil_type
        )
        market_future = pool.submit(_safe_call, get_live_market_indicators)

        weather_data, nasa_parameters = nasa_daily_future.result()
        climatology = climatology_future.result() or _build_nasa_climatology_fallback(latitude)
        forecast_payload = forecast_future.result()
        if not forecast_payload:
            forecast_payload = _build_forecast_fallback(latitude, climatology)
        outlook = outlook_future.result()
        soilgrids_data = soilgrids_future.result() or {
            **_soilgrids_fallback_by_region(governorate, soil_type),
            "source": "Regional Soil Baseline Fallback",
            "is_fallback": True,
        }
        market_indicators = market_future.result() or {
            "usd_egp_rate": FALLBACK_USD_EGP,
            "baseline_usd_egp": BASELINE_USD_EGP,
            "fx_ratio": round(FALLBACK_USD_EGP / BASELINE_USD_EGP, 4),
            "fertilizer_global_index": 1.04,
            "cost_index_factor": 1.0,
            "price_index_factor": 1.0,
            "source": "Baseline Market Fallback",
            "updated_utc": date.today().isoformat(),
            "is_fallback": True,
        }

    open_meteo_elev = forecast_payload.get("elevation") if forecast_payload else None
    elevation_data = get_elevation_srtm(
        latitude, longitude, governorate=governorate, open_meteo_elevation=open_meteo_elev
    )

    return (
        weather_data,
        nasa_parameters,
        climatology,
        forecast_payload,
        outlook,
        soilgrids_data,
        elevation_data,
        market_indicators,
    )


@app.get("/api/soil")
def get_soil_endpoint(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    governorate: Optional[str] = Query(default=None),
    language: str = Query(default="ar"),
):
    sg = get_soilgrids_data(latitude, longitude, governorate=governorate)
    elev = get_elevation_srtm(latitude, longitude, governorate=governorate)
    inferred_texture = _classify_soil_texture_from_percentages(sg.get("clay_pct"), sg.get("sand_pct"))
   
    return {
        "success": True,
        "latitude": latitude,
        "longitude": longitude,
        "soilgrids": {
            **sg,
            "inferred_texture_ar": inferred_texture,
            "inferred_texture_en": LABELS_EN.get(inferred_texture, inferred_texture),
            
       
    },
 "elevation": elev,
    }
@app.post("/api/analyze")
def analyze_farm(farm: FarmData):

    # Step 1: Validate farmer answers

    validate_farm_data(farm)

    lang = get_language(farm)

    # Step 2: Select farm coordinates

    location = select_farm_location(farm)

    latitude = location["latitude"]

    longitude = location["longitude"]

    # Step 3: Fetch NASA POWER, Climatology, SoilGrids, SRTM Elevation, Open-Meteo & Live FX in parallel

    (
        weather_data,
        nasa_parameters,
        climatology,
        forecast_payload,
        outlook,
        soilgrids_data,
        elevation_data,
        market_indicators,
    ) = _fetch_farm_environmental_data(
        latitude, longitude, governorate=farm.governorate, soil_type=farm.soil_type
    )

    # Step 4: Build Hybrid Soil Profile & Weather Summary

    soil_profile = build_hybrid_soil_profile(farm, soilgrids_data, lang)

    weather_summary = calculate_weather_summary(weather_data)

    season = build_season(farm.planting_month, weather_summary, climatology)

    forecast_days = (forecast_payload or {}).get("days") or []

    # Step 5: Create assessment from answers, live NASA weather, hybrid soil & SRTM elevation

    assessment, primary_evaluation, ctx = create_initial_assessment(
        farm,
        weather_summary,
        nasa_parameters["end"],
        lang,
        climatology,
        season=season,
        soil_profile=soil_profile,
        elevation_data=elevation_data,
    )

    # Step 5B: Forecast section (with FAO ET0)

    forecast = build_forecast_section(
        farm, forecast_payload, outlook, climatology, lang
    )

    # Step 5C: Live NASA Water Demand, Dynamic Economics & 16-Day Schedule + Crop Calendar

    live_water_demand = calculate_live_crop_water_demand(
        farm.crop, farm, weather_summary, season, forecast_days,
        soil_profile, elevation_data, lang
    )

    economics_and_water = calculate_economics_and_water(
        farm.crop, farm, lang,
        live_water=live_water_demand,
        market_indicators=market_indicators,
        apply_manual_overrides=True,
    )

    crop_calendar = build_crop_calendar(
        farm.crop, farm, lang,
        soil_profile=soil_profile,
        live_water=live_water_demand,
        forecast_days=forecast_days,
    )

    irrigation_schedule_16_days = build_16_day_irrigation_schedule(
        farm.crop, farm, forecast_days, weather_summary, soil_profile, lang
    )

    # Step 5D: Optional side-by-side comparison with a second crop

    crop_comparison = None
    if farm.compare_crop is not None:
        if farm.crop != "مش متأكد":
            crop_comparison = compare_two_crops(
                farm.crop, farm.compare_crop, farm, ctx, lang,
                forecast_days=forecast_days, market_indicators=market_indicators
            )
        elif assessment.get("alternative_crops"):
            top_recommended = assessment["alternative_crops"][0]["crop"]
            if top_recommended != farm.compare_crop:
                crop_comparison = compare_two_crops(
                    top_recommended, farm.compare_crop, farm, ctx, lang,
                    forecast_days=forecast_days, market_indicators=market_indicators
                )
                    # Step 5D-2: Crop Rotation Strategies

    target_crop_for_rot = (
        farm.crop
        if farm.crop != "مش متأكد"
        else (
            assessment["alternative_crops"][0]["crop"]
            if assessment.get("alternative_crops")
            else "القمح"
        )
    )

    # Fetch real MODIS NDVI data for the selected farm coordinates
    ndvi_data = get_ndvi_series(
        latitude,
        longitude,
        lang
    )

    # Estimate low vegetation cover duration from NDVI
    fallow_days = calculate_fallow_days_from_ndvi(
        ndvi_data
    )

    # Generate crop rotation strategies using NDVI data
    rotation_strategies = [
        model_to_dict(s)
        for s in generate_rotation_strategies(
            farm.previous_crop or "غير معروف",
            target_crop_for_rot,
            fallow_days_override=fallow_days
        )
    ]

    # Step 5E: Transparent methodology

    # Step 5E: Transparent methodology

    methodology = build_methodology(lang, evaluation=primary_evaluation)

    # Step 6: Build limitations

    limitations = build_limitations(
        location,
        weather_summary,
        farm,
        lang,
        season,
        forecast,
        soil_profile=soil_profile,
    )

    # Step 7: Prepare farm information

    farm_information = model_to_dict(farm)

    month_climatology = (
        climatology.get(farm.planting_month)
        if climatology else None
    )

    # Step 8: Return complete response

    return {

        "success": True,

        "application": "N.A.D.A",

        "version": APP_VERSION,

        "farm": farm_information,

        "location": location,

        "elevation": elevation_data,

        "soil_profile": soil_profile,

        "market_indicators": market_indicators,

        "live_water_demand": live_water_demand,

        "irrigation_schedule_16_days": irrigation_schedule_16_days,

        "climate": {

            "source": "NASA POWER",

            "period_start": nasa_parameters["start"],

            "period_end": nasa_parameters["end"],

            "period_days": WEATHER_PERIOD_DAYS,

            "summary": weather_summary,

            "planting_month_climatology": (
                {
                    "source": "NASA POWER climatology",
                    "month": farm.planting_month,
                    **month_climatology,
                }
                if month_climatology else None
            ),

        },

   "assessment": assessment,

        "economics_and_water": economics_and_water,

        "crop_calendar": crop_calendar,

        "crop_comparison": crop_comparison,

        "rotation_strategies": rotation_strategies,

        "methodology": methodology,

        "forecast": forecast,

        "limitations": limitations,

    }

# ============================================================
# 24B. DEDICATED SIDE-BY-SIDE CROP COMPARISON ENDPOINT
# ============================================================

@app.post("/api/compare")
def compare_crops_endpoint(request: CompareRequest):

    validate_farm_data(request)

    if request.crop == "مش متأكد":
        raise HTTPException(
            status_code=422,
            detail="Please select two specific crops to compare."
        )

    lang = get_language(request)
    location = select_farm_location(request)
    latitude = location["latitude"]
    longitude = location["longitude"]

    (
        weather_data,
        nasa_parameters,
        climatology,
        forecast_payload,
        _,
        soilgrids_data,
        elevation_data,
        market_indicators,
    ) = _fetch_farm_environmental_data(
        latitude, longitude, governorate=request.governorate, soil_type=request.soil_type
    )

    soil_profile = build_hybrid_soil_profile(request, soilgrids_data, lang)
    weather_summary = calculate_weather_summary(weather_data)
    season = build_season(request.planting_month, weather_summary, climatology)
    forecast_days = (forecast_payload or {}).get("days") or []

    ctx = build_climate_context(
        weather_summary, nasa_parameters["end"], climatology, request.planting_month,
        season=season, soil_profile=soil_profile, elevation_data=elevation_data
    )

    comparison = compare_two_crops(
        request.crop, request.compare_crop, request, ctx, lang,
        forecast_days=forecast_days, market_indicators=market_indicators
    )

    return {
        "success": True,
        "application": "N.A.D.A",
        "version": APP_VERSION,
        "location": location,
        "elevation": elevation_data,
        "soil_profile": soil_profile,
        "market_indicators": market_indicators,
        "planting_month": request.planting_month,
        "climate_period": {
            "start": nasa_parameters["start"],
            "end": nasa_parameters["end"],
            "planting_month_climatology": (
                climatology.get(request.planting_month) if climatology else None
            ),
        },
        "comparison": comparison,
        "methodology": build_methodology(lang),
    }


# ============================================================
# 24C. TRANSPARENT METHODOLOGY ENDPOINT
# ============================================================

@app.get("/api/methodology")
def get_methodology(language: str = Query(default="ar")):

    lang = normalize_backend_lang(language)

    return {
        "success": True,
        "application": "N.A.D.A",
        "version": APP_VERSION,
        "methodology": build_methodology(lang),
    }

# ============================================================
# 13G. NDVI VEGETATION INDEX (NASA MODIS MOD13Q1 via ORNL DAAC)
# الصق هذا الجزء في NASA.py قبل القسم "25. HEALTH CHECK ENDPOINT"
# لا يحتاج أي مكتبات جديدة ولا مفتاح API.
# ============================================================

NDVI_URL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
NDVI_BAND = "250m_16_days_NDVI"
NDVI_CACHE_TTL = 12 * 3600
NDVI_TIMEOUT = 15.0          # خادم ORNL أبطأ من باقي المصادر
NDVI_PERIODS = 10           # خادم ORNL يسمح بحد أقصى 10 فترات في الطلب الواحد (نحو 5 شهور)


def _modis_date(d):
    return f"A{d.year}{d.timetuple().tm_yday:03d}"


def _ndvi_label(value, lang):
    if value is None:
        return "unknown", L(lang, "غير متاح", "Unavailable")
    if value >= 0.6:
        return "excellent", L(lang, "ممتازة", "Excellent")
    if value >= 0.4:
        return "good", L(lang, "جيدة", "Good")
    if value >= 0.2:
        return "moderate", L(lang, "متوسطة", "Moderate")
    return "poor", L(lang, "ضعيفة", "Poor")


def get_ndvi_series(latitude, longitude, lang="ar"):

    cache_key = ("ndvi_modis", *_coord_cache_key(latitude, longitude))
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    end_date = date.today() - timedelta(days=32)  # آخر مركّبات MODIS بتتأخر أسابيع
    start_date = end_date - timedelta(days=16 * (NDVI_PERIODS - 1))

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "band": NDVI_BAND,
        "startDate": _modis_date(start_date),
        "endDate": _modis_date(end_date),
        "kmAboveBelow": 0,
        "kmLeftRight": 0,
    }

    session = _get_http_session()
    last_error = "unknown"

    try:
        payload = None
        for attempt in range(2):  # محاولتان لأن خادم ORNL بطيء أحياناً
            try:
                response = session.get(
                    NDVI_URL,
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=NDVI_TIMEOUT,
                )
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning("MODIS NDVI %s", last_error)
                    continue
                payload = response.json()
                break
            except requests.exceptions.RequestException as req_err:
                last_error = f"{type(req_err).__name__}: {req_err}"
                logger.warning("MODIS NDVI attempt %s failed: %s", attempt + 1, last_error)

        if payload is None:
            raise ValueError(last_error)

        scale = _safe_float(payload.get("scale")) or 0.0001
        points = []

        for item in payload.get("subset", []):
            values = item.get("data") or []
            raw = _safe_float(values[0]) if values else None
            # القيم الأقل من -2000 هي قيم "لا بيانات" في MODIS
            if raw is None or raw <= -2000:
                continue
            ndvi = round(max(-0.2, min(1.0, raw * scale)), 3)
            points.append({
                "date": str(item.get("calendar_date")),
                "ndvi": ndvi,
            })

        if not points:
            raise ValueError("No valid NDVI points")

        points.sort(key=lambda p: p["date"])
        latest = points[-1]["ndvi"]
        status_code, status_text = _ndvi_label(latest, lang)

        result = {
            "available": True,
            "source": "NASA MODIS MOD13Q1 (250m, 16-day composite) via ORNL DAAC",
            "latest_ndvi": latest,
            "latest_date": points[-1]["date"],
            "status": status_code,
            "status_text": status_text,
            "average_ndvi": round(mean(p["ndvi"] for p in points), 3),
            "series": points,
            "is_fallback": False,
        }
        _DATA_CACHE.set(cache_key, result, NDVI_CACHE_TTL)
        return result

    except (
        requests.exceptions.RequestException,
        ValueError, KeyError, TypeError, AttributeError
    ) as error:
        logger.warning("MODIS NDVI request failed (%s)", error)
        error_text = str(error)[:300]
        # لا نخترع أرقاماً: نرجع حالة "غير متاح" بصراحة
        return {
            "available": False,
            "source": "NASA MODIS MOD13Q1 via ORNL DAAC",
            "latest_ndvi": None,
            "latest_date": None,
            "status": "unknown",
            "status_text": L(lang, "غير متاح حالياً", "Currently unavailable"),
            "average_ndvi": None,
            "series": [],
            "is_fallback": True,
            "error": error_text,
        }


@app.get("/api/ndvi")
def get_ndvi_endpoint(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    language: str = Query(default="ar"),
):
    lang = normalize_backend_lang(language)
    return {
        "success": True,
        "latitude": latitude,
        "longitude": longitude,
        "ndvi": get_ndvi_series(latitude, longitude, lang),
    }


# ============================================================
# 25. HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/api/health")
def health_check():

    return {

        "status": "running",

        "application": "N.A.D.A",

        "version": APP_VERSION,

    }


# ============================================================
# 26. ROOT ENDPOINT (SERVING index.html FOR VERCEL SERVERLESS)
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():

    html_path = os.path.join(os.path.dirname(__file__), "index.html")

    if not os.path.exists(html_path):
        return HTMLResponse(
            content=(
                "<!DOCTYPE html><html lang='ar' dir='rtl'><head><meta charset='UTF-8'>"
                "<title>N.A.D.A</title></head><body style='font-family:sans-serif;padding:2rem;'>"
                "<h2>N.A.D.A API is running</h2>"
                "<p>ملف index.html غير موجود بجانب NASA.py. يرجى وضعه في نفس المجلد.</p>"
                "</body></html>"
            ),
            status_code=200,
        )

    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    return HTMLResponse(content=html_content, status_code=200)


# ============================================================
# 26B. ASK AI AGRONOMIC ADVISOR (GEMINI API - ALL WORLD LANGUAGES)
# ============================================================

class AskAIRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    farm_context: dict = Field(default_factory=dict)
    language: Optional[str] = "ar"


def _load_gemini_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    alt_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if alt_key:
        return alt_key
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    clean_line = line.lstrip("\ufeff").strip()
                    if clean_line.startswith("GEMINI_API_KEY="):
                        return clean_line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


@app.post("/api/ask-ai")
def ask_ai_advisor(req: AskAIRequest):
    import json

    api_key = _load_gemini_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="مفتاح GEMINI_API_KEY غير موجود! أضفه في Environment Variables على Vercel (ثم اعمل Redeploy) أو في ملف .env."
        )

    custom_model = os.environ.get("GEMINI_MODEL", "").strip()
    candidate_models = [custom_model] if custom_model else [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-2.5-flash",
    ]

    context_json = json.dumps(req.farm_context, ensure_ascii=False)
    req_lang = (req.language or "ar").strip()

    if req_lang.lower().startswith("ar"):
        prompt = (
            "أنت مرشد زراعي ذكي خبير في الزراعة المصرية وبيانات أقمار ناسا (N.A.D.A). "
            "أجب باللهجة المصرية البسيطة الممزوجة بالعربية الفصحى السهلة الواضحة للفلاح. "
            "التزم بأن تكون إجابتك عملية ومباشرة ومختصرة (في حدود 4 إلى 6 أسطر أو نقاط واضحة) "
            "ومبنية على بيانات الأرض التالية فقط:\n\n"
            f"بيانات الأرض والتحليل الحالي:\n{context_json}\n\n"
            f"سؤال المزارع:\n{req.question.strip()}"
        )
    else:
        prompt = (
            "You are N.A.D.A, an expert smart agricultural advisor for farming and NASA satellite data. "
            f"You MUST write your entire reply in the language corresponding to code/name: '{req_lang}'. "
            "Keep your reply clear, practical, simple for a farmer, and concise (4 to 6 short lines or bullet points), "
            "strictly grounded in the following farm data:\n\n"
            f"Current farm & analysis data:\n{context_json}\n\n"
            f"Farmer question:\n{req.question.strip()}"
        )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1024,
        },
    }

    last_error_msg = "تعذر الاتصال بـ Gemini API."

    for model_name in candidate_models:
        gemini_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent"
        )
        try:
            resp = requests.post(
                gemini_url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                json=payload,
                timeout=8,
            )

            if resp.status_code == 404:
                last_error_msg = f"الموديل {model_name} غير متاح (404)."
                continue

            if not resp.ok:
                try:
                    err_json = resp.json()
                    google_msg = err_json.get("error", {}).get("message", resp.text)
                except Exception:
                    google_msg = resp.text[:200]
                last_error_msg = f"خطأ Gemini ({resp.status_code}): {google_msg}"
                logger.warning("Gemini API error (%s): %s", model_name, last_error_msg)
                break

            data = resp.json()
            candidates = data.get("candidates") or []
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []

            text_parts = [p.get("text", "") for p in parts if not p.get("thought") and p.get("text")]
            if not text_parts:
                text_parts = [p.get("text", "") for p in parts if p.get("text")]

            answer = "".join(text_parts).strip()
            if not answer:
                finish_reason = candidates[0].get("finishReason") if candidates else "UNKNOWN"
                last_error_msg = f"رد فارغ من {model_name} (finishReason: {finish_reason})"
                continue

            return {
                "success": True,
                "model": model_name,
                "answer": answer,
            }

        except requests.exceptions.Timeout:
            last_error_msg = "انتهت مهلة الاتصال بـ Gemini (8 ثوانٍ). حاول مرة أخرى."
            break
        except requests.exceptions.RequestException as error:
            last_error_msg = f"فشل الاتصال الشبكي بـ Gemini: {error}"
            break

    raise HTTPException(status_code=502, detail=last_error_msg)


# ============================================================
# 27. RUN SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(

        "NASA:app",

        host="127.0.0.1",

        port=8000,

        reload=True

    )
