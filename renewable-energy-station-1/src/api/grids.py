from pathlib import Path
from flask import Blueprint, render_template
from src.models.grid_factory import GridFactory

bp = Blueprint("grids", __name__, url_prefix="/grids")

config_path = Path("/app/config/grids.json")
grid_factory = GridFactory(str(config_path))


@bp.route("/")
def list_grids():
    try:
        available_grids = grid_factory.get_available_grids()
        return render_template(
            "grids.html",
            grids=available_grids,
            grid_factory=grid_factory,
        )
    except Exception as e:
        print(f"Error loading grids: {e}")
        return render_template(
            "grids.html",
            grids=[],
            grid_factory=grid_factory,
            error=str(e),
        )


@bp.route("/<grid_type>")
def grid_detail(grid_type):
    grid = grid_factory.create_grid(grid_type)
    return render_template("grid_detail.html", grid=grid, grid_type=grid_type)
