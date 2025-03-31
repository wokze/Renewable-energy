# src/api/photovoltaics.py
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify
from src.models.photovoltaic_factory import PhotovoltaicFactory

bp = Blueprint("photovoltaics", __name__, url_prefix="/photovoltaics")

config_path = Path("/app/config/pv.json")
pv_factory = PhotovoltaicFactory(str(config_path))


@bp.route("/")
def list_pvs():
    try:
        available_pvs = pv_factory.get_available_pvs()
        print(f"Found PVs: {available_pvs}")  # Debug line
        return render_template(
            "photovoltaics.html",
            pvs=available_pvs,
            pv_factory=pv_factory,
        )
    except Exception as e:
        print(f"Error loading PVs: {e}")
        return render_template(
            "photovoltaics.html",
            pvs=[],
            pv_factory=pv_factory,
            error=str(e),
        )


@bp.route("/<pv_type>")
def pv_detail(pv_type):
    pv = pv_factory.create_pv(pv_type)
    return render_template("pv_detail.html", pv=pv, pv_type=pv_type)


@bp.route("/<pv_type>/forecast")
def pv_forecast(pv_type):
    pv = pv_factory.create_pv(pv_type)

    # Get parameters from query string
    latitude = request.args.get("lat", type=float, default=40.0)
    longitude = request.args.get("lon", type=float, default=20.0)
    angle = request.args.get("angle", type=float, default=35.0)
    azimuth = request.args.get("azimuth", type=float, default=0.0)

    try:
        forecast_data = pv.get_pvgis_forecast(
            latitude=latitude,
            longitude=longitude,
            installation_angle=angle,
            azimuth=azimuth,
        )

        # For AJAX requests return JSON
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(forecast_data)

        # Otherwise render the template
        return render_template(
            "pv_forecast.html",
            pv=pv,
            pv_type=pv_type,
            forecast_data=forecast_data,
            latitude=latitude,
            longitude=longitude,
            angle=angle,
            azimuth=azimuth,
        )
    except Exception as e:
        error_msg = str(e)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": error_msg}), 400
        return render_template("error.html", error=error_msg)
