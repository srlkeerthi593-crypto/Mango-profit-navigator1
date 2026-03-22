# ============================================================
# 🥭 FARMER PROFIT INTELLIGENCE SYSTEM - FLASK VERSION
# Smart Mango Marketing Decision Engine
# Developed by: S.R.L. Keerthi
# ============================================================

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os, requests

app = Flask(__name__)

# ── LOAD DATA ────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

def p(f):
    return os.path.join(BASE, f)

villages   = pd.read_csv(p("Village data.csv"))
prices     = pd.read_csv(p("cleaned_price_data.csv"))
geo        = pd.read_csv(p("cleaned_geo_locations.csv"))
processing = pd.read_csv(p("cleaned_processing_facilities.csv"))
pulp       = pd.read_csv(p("Pulp_units_merged_lat_long.csv"))
pickle_u   = pd.read_csv(p("cleaned_pickle_units.csv"))
local_exp  = pd.read_csv(p("cleaned_local_export.csv"))
abroad_exp = pd.read_csv(p("cleaned_abroad_export.csv"))

for df in [villages, prices, geo, processing, pulp, pickle_u, local_exp, abroad_exp]:
    df.columns = df.columns.str.strip().str.lower()

# ── HELPERS ──────────────────────────────────────────────────
def haversine(la1, lo1, la2, lo2):
    R = 6371
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

VARIETY_ACCEPT = {
    "Mandi":         ["Banganapalli","Totapuri","Neelam","Rasalu"],
    "Processing":    ["Totapuri","Neelam"],
    "Pulp":          ["Totapuri"],
    "Pickle":        ["Totapuri","Rasalu"],
    "Local Export":  ["Banganapalli"],
    "Abroad Export": ["Banganapalli"],
}
MARGIN_MAP = {
    "Mandi":0, "Processing":0.03, "Pulp":0.04,
    "Pickle":0.025, "Local Export":0.05, "Abroad Export":0.07
}

def compute_top10(v_lat, v_lon, base_price, qty, variety):
    sources = {
        "Mandi":        (prices,     "place",           "lat",      "long"),
        "Processing":   (processing, "facility_name",   "latitude", "longitude"),
        "Pulp":         (pulp,       "facility name",   "latitude", "longitude"),
        "Pickle":       (pickle_u,   "firm_name",       "latitude", "longitude"),
        "Local Export": (local_exp,  "hub_/_firm_name", "latitude", "longitude"),
        "Abroad Export":(abroad_exp, "place_name",      "latitude", "longitude"),
    }
    results = []
    for cat, (df, name_col, lat_col, lon_col) in sources.items():
        if variety not in VARIETY_ACCEPT[cat]:
            continue
        margin = MARGIN_MAP[cat]
        df_c = df.copy()
        df_c.columns = df_c.columns.str.strip().str.lower()
        for _, row in df_c.iterrows():
            try:
                la = float(row[lat_col.lower()])
                lo = float(row[lon_col.lower()])
                nm = str(row[name_col.lower()])
            except Exception:
                continue
            if np.isnan(la) or np.isnan(lo):
                continue
            dist      = haversine(v_lat, v_lon, la, lo)
            transport = dist * 12 * qty
            revenue   = base_price * (1 + margin) * 100 * qty
            net       = revenue - transport
            results.append({
                "category": cat,
                "name": nm,
                "distance_km": round(dist, 1),
                "revenue": round(revenue),
                "transport": round(transport),
                "net_profit": round(net),
                "lat": la,
                "lon": lo,
            })

    seen = set()
    deduped = []
    for r in results:
        k = r["name"] + "|" + r["category"]
        if k not in seen:
            seen.add(k)
            deduped.append(r)

    deduped.sort(key=lambda x: x["net_profit"], reverse=True)
    top10 = deduped[:10]
    for i, r in enumerate(top10):
        r["rank"] = i + 1
    return top10

# ── ROUTES ───────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/villages")
def get_villages():
    result = {}
    for _, row in villages.iterrows():
        mandal = str(row.get("mandal", "")).strip()
        gp     = str(row.get("gram panchayat", "")).strip()
        if mandal not in result:
            result[mandal] = []
        result[mandal].append(gp)
    for m in result:
        result[m] = sorted(result[m])
    return jsonify(result)

@app.route("/api/prices")
def get_prices():
    data = []
    for _, row in prices.iterrows():
        data.append({
            "place":     str(row.get("place", "")),
            "today":     float(row.get("today_price(rs/kg)", 0)),
            "yesterday": float(row.get("yesterday_price(rs/kg)", 0)),
        })
    return jsonify(data)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    body    = request.json
    village = body.get("village", "")
    variety = body.get("variety", "Banganapalli")
    qty     = int(body.get("qty", 10))

    # Find village coordinates
    v_row = villages[villages["gram panchayat"] == village]
    if v_row.empty:
        return jsonify({"error": "Village not found"}), 404

    v_lat = float(v_row.iloc[0]["latitude"])
    v_lon = float(v_row.iloc[0]["longitude"])

    # Find nearest mandi base price
    base_price = 30
    nearest_dist = float("inf")
    for _, pr in prices.iterrows():
        try:
            d = haversine(v_lat, v_lon, float(pr["lat"]), float(pr["long"]))
            if d < nearest_dist:
                nearest_dist = d
                base_price = float(pr["today_price(rs/kg)"])
        except Exception:
            continue

    top10 = compute_top10(v_lat, v_lon, base_price, qty, variety)

    return jsonify({
        "village": village,
        "v_lat":   v_lat,
        "v_lon":   v_lon,
        "base_price": base_price,
        "qty":     qty,
        "variety": variety,
        "results": top10,
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
