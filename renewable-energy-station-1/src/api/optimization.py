# src/api/optimization.py
from flask import Blueprint, request, jsonify, render_template, abort
from src.config.database import SessionLocal
from src.schemas.scenario import Scenario
from ..algorithms.tilos import TILOSOptimizer

bp = Blueprint("optimization", __name__, url_prefix="/optimization")


@bp.route("/<int:scenario_id>/optimize", methods=["GET", "POST"])
def optimize_scenario(scenario_id):
    db = SessionLocal()
    # Replace first_or_404() with first() and manual check
    scenario = db.query(Scenario).filter_by(id=scenario_id).first()

    # Handle 404 manually
    if not scenario:
        abort(404, description=f"Scenario with id {scenario_id} not found")

    if request.method == "POST":
        # Get forecast data from request
        data = request.json
        solar_forecast = data.get("solar_forecast", [0] * 24)
        wind_forecast = data.get("wind_forecast", [0] * 24)
        demand_forecast = data.get("demand_forecast", [0] * 24)

        # Run optimization
        optimizer = TILOSOptimizer(scenario)
        result = optimizer.optimize(solar_forecast, wind_forecast, demand_forecast)

        return jsonify(result)

    # GET request shows optimization form
    return render_template("optimization.html", scenario=scenario)
