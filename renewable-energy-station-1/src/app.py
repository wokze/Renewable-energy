from datetime import datetime
from flask import Flask, render_template
from flask_restful import Api
from src.models.battery import BatteryFactory
from src.models.photovoltaic_factory import PhotovoltaicFactory
from src.models.grid_factory import GridFactory
from src.models.turbine_factory import TurbineFactory
from src.models.grid import Grid
from src.config.database import init_db, SessionLocal
from src.api import (
    batteries,
    scenarios,
    photovoltaics,
    turbines,
    grids,
    optimization,
)  # Import blueprints
from src.schemas import Base, Scenario, Location  # Import models


def create_app():
    app = Flask(__name__)
    api = Api(app)

    # Initialize database first
    init_db()

    # Register blueprints
    app.register_blueprint(batteries.bp, url_prefix="/batteries")
    app.register_blueprint(scenarios.bp, url_prefix="/scenarios")
    app.register_blueprint(photovoltaics.bp, url_prefix="/photovoltaics")
    app.register_blueprint(turbines.bp, url_prefix="/turbines")
    app.register_blueprint(grids.bp, url_prefix="/grids")
    app.register_blueprint(optimization.bp)  # Add this line

    # Initialize factories
    battery_factory = BatteryFactory("/app/config/battery_configs.json")
    pv_factory = PhotovoltaicFactory("/app/config/pv.json")
    turbine_factory = TurbineFactory("/app/config/turbines.json")
    grid_factory = GridFactory("/app/config/grids.json")

    try:
        # Initialize default components
        available_pvs = pv_factory.get_available_pvs()
        app.solar_panels = []
        if available_pvs:
            # Create one instance of each available PV type
            for pv_type in available_pvs:
                pv = pv_factory.create_pv(pv_type)
                if pv:
                    app.solar_panels.append({"model_type": pv_type, "quantity": 1})

        # Initialize wind turbines
        available_turbines = turbine_factory.get_available_turbines()
        app.wind_turbines = []
        if available_turbines:
            for turbine_type in available_turbines:
                turbine = turbine_factory.create_turbine(turbine_type)
                if turbine:
                    app.wind_turbines.append(
                        {
                            "turbine_type": turbine_type,
                            "quantity": 1,
                            "model": turbine.model_name,
                            "capacity": turbine.capacity,
                        }
                    )

        app.grid = grid_factory.create_grid("main_grid")

    except Exception as e:
        print(f"Error initializing components: {e}")
        app.solar_panels = []
        app.wind_turbines = []
        app.grid = None

    @app.context_processor
    def utility_processor():
        return dict(current_year=datetime.utcnow().year)

    @app.route("/")
    def index():
        try:
            return render_template(
                "dashboard.html",
                # Battery info
                battery_factory=battery_factory,
                batteries=battery_factory.get_available_batteries(),
                # Solar info
                pv_factory=pv_factory,
                solar_panels=app.solar_panels,
                solar_status={
                    "irradiance": 1000,  # Example value
                    "temperature": 25,  # Example value
                },
                # Wind info
                turbine_factory=turbine_factory,
                wind_turbines=app.wind_turbines,  # Use the initialized wind_turbines
                wind_status={"wind_speed": 10.0},  # Example value
                # Grid info
                grid_status=app.grid.get_status() if app.grid else None,
            )
        except Exception as e:
            print(f"Error loading dashboard: {e}")
            return render_template("error.html", error=str(e))

    return app


app = create_app()
