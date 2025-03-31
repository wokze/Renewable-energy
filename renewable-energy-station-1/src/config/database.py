import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.schemas import Base  # Import Base from schemas instead of creating new one

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@db:5432/renewable_energy"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize the database"""
    try:
        print("Starting database initialization...")

        # Drop materialized view if it exists
        with engine.connect() as conn:
            conn.execute(
                text("DROP MATERIALIZED VIEW IF EXISTS battery_metrics_daily CASCADE")
            )
            conn.commit()
            print("Dropped existing materialized views")

        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        print("Dropped all existing tables")

        # Create all tables defined in schemas
        Base.metadata.create_all(bind=engine)
        print("Created all tables from schemas")

        # Create TimescaleDB hypertable
        with engine.connect() as conn:
            try:
                conn.execute(
                    text(
                        """
                        SELECT create_hypertable('battery_metrics', 'time', 
                            if_not_exists => TRUE,
                            chunk_time_interval => INTERVAL '1 day'
                        )
                    """
                    )
                )
                conn.commit()
                print("Created TimescaleDB hypertable")
            except Exception as e:
                print(f"Warning: Could not create hypertable: {e}")
                # Continue even if hypertable creation fails

        print("Database initialization completed successfully")

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
