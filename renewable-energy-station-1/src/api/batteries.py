from pathlib import Path
from flask import Blueprint, render_template, redirect, url_for, request
from src.models.battery import BatteryFactory

bp = Blueprint("batteries", __name__, url_prefix="/batteries")

# Get absolute path to config file
config_path = Path("/app/config/battery_configs.json")


battery_factory = BatteryFactory(str(config_path))


@bp.route("/")
def list_batteries():
    try:
        available_batteries = battery_factory.get_available_batteries()
        print(f"Found batteries: {available_batteries}")  # Debug line
        return render_template(
            "batteries.html",
            batteries=available_batteries,
            battery_factory=battery_factory,
        )
    except Exception as e:
        print(f"Error loading batteries: {e}")  # Debug line
        return render_template(
            "batteries.html",
            batteries=[],
            battery_factory=battery_factory,
            error=str(e),
        )


@bp.route("/<battery_type>")
def battery_detail(battery_type):
    battery = battery_factory.create_battery(battery_type)
    return render_template(
        "battery_detail.html", battery=battery, battery_type=battery_type
    )


@bp.route("/update/<battery_type>", methods=["POST"])
def update_battery(battery_type):
    quantity = request.form.get("quantity", type=int)
    status = request.form.get("status")
    # Update database
    return redirect(url_for("batteries.battery_detail", battery_type=battery_type))
