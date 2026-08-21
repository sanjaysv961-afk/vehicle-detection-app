from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Simple rule-based ML-style predictor ──────────────────────────────────────
# In a real project you'd load a trained sklearn model here:
#   import joblib
#   model = joblib.load("attendance_model.pkl")

def predict_attendance(total: int, attended: int):
    if total <= 0:
        return None, None, None, None, None

    pct = (attended / total) * 100

    # Risk classification (mirrors a trained Decision-Tree boundary)
    if pct >= 85:
        risk = "Low Risk"
        color = "green"
        advice = "Excellent attendance! Keep it up."
    elif pct >= 75:
        risk = "Moderate Risk"
        color = "orange"
        needed = max(0, round((0.75 * total - attended) / (1 - 0.75)))
        advice = f"You need {needed} more classes to stay comfortably safe."
    else:
        # Calculate classes needed to reach 75%
        needed = max(0, round((0.75 * total - attended) / (1 - 0.75)))
        risk = "High Risk"
        color = "red"
        advice = f"Attend next {needed} consecutive classes to reach 75%."

    # How many classes can still be safely missed?
    can_miss = max(0, int(attended - 0.75 * total))

    return round(pct, 2), risk, color, advice, can_miss


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        total    = int(data["total"])
        attended = int(data["attended"])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    if attended > total:
        return jsonify({"error": "Attended cannot exceed total classes"}), 400

    pct, risk, color, advice, can_miss = predict_attendance(total, attended)

    if pct is None:
        return jsonify({"error": "Total classes must be greater than 0"}), 400

    return jsonify({
        "percentage": pct,
        "risk":       risk,
        "color":      color,
        "advice":     advice,
        "can_miss":   can_miss,
        "result":     f"{pct}% attendance — {risk}"
    })


@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    """Accept a list of students and return predictions for each."""
    students = request.get_json(force=True)
    results  = []
    for s in students:
        try:
            total    = int(s["total"])
            attended = int(s["attended"])
            name     = s.get("name", "Unknown")
            pct, risk, color, advice, can_miss = predict_attendance(total, attended)
            results.append({
                "name":       name,
                "percentage": pct,
                "risk":       risk,
                "color":      color,
                "advice":     advice,
                "can_miss":   can_miss,
            })
        except Exception:
            results.append({"name": s.get("name", "?"), "error": "Invalid data"})
    return jsonify(results)


@app.route("/stats", methods=["POST"])
def stats():
    """Return simple descriptive statistics for a class."""
    students = request.get_json(force=True)
    pcts = []
    for s in students:
        try:
            pcts.append(int(s["attended"]) / int(s["total"]) * 100)
        except Exception:
            pass
    if not pcts:
        return jsonify({"error": "No valid data"}), 400
    arr = np.array(pcts)
    return jsonify({
        "mean":   round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "min":    round(float(arr.min()), 2),
        "max":    round(float(arr.max()), 2),
        "std":    round(float(arr.std()), 2),
        "at_risk_count": int((arr < 75).sum()),
        "total_students": len(arr),
    })


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
