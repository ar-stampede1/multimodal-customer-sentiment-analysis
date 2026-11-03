
from sqlalchemy import text

from backend.app.database.connection import engine

try:
    with engine.connect() as connection:

        result = connection.execute(
            text("SELECT version();")
        )

        print("DATABASE CONNECTION SUCCESSFUL")
        print("--------------------------------")

        print(result.scalar())

        print("\nDatabase:", engine.url.database)
        print("Host:", engine.url.host)
        print("Port:", engine.url.port)

except Exception as error:

    print("DATABASE CONNECTION FAILED")
    print(error)

    raise
