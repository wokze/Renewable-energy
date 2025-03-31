from pathlib import Path
from flask import Blueprint, render_template
from src.models.turbine_factory import TurbineFactory

bp = Blueprint("turbines", __name__, url_prefix="/turbines")

config_path = Path("/app/config/turbines.json")
turbine_factory = TurbineFactory(str(config_path))


@bp.route("/")
def list_turbines():
    try:
        available_turbines = turbine_factory.get_available_turbines()
        print(f"Found turbines: {available_turbines}")  # Debug line
        return render_template(
            "turbines.html",
            turbines=available_turbines,
            turbine_factory=turbine_factory,
        )
    except Exception as e:
        print(f"Error loading turbines: {e}")
        return render_template(
            "turbines.html",
            turbines=[],
            turbine_factory=turbine_factory,
            error=str(e),
        )


@bp.route("/<turbine_type>")
def turbine_detail(turbine_type):
    turbine = turbine_factory.create_turbine(turbine_type)
    return render_template(
        "turbine_detail.html", turbine=turbine, turbine_type=turbine_type
    )
