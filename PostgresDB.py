import psycopg2

class PostgresDB:
    def __init__(self, password, host='localhost', database='mwdb_project', user='postgres', port=5432):
        self.host = host
        self.database = database
        self.user = user
        self.port = port
        self.password = password

    def connect(self):
        conn = None
        try:
            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(host=self.host, database=self.database,
                                    user=self.user, password=self.password, port=self.port)

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        return conn

