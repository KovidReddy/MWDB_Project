import psycopg2
import json
class PostgresDB:
    # Fetch Config file
    def config(self):
        try:
            with open('config.json') as f:
                js = json.load(f)
                host = js['host']
                database = js['database']
                user = js['user']
                port = js['port']
                password = js['password']

        except Exception as error:
            print(error)
            exit(-1)

        return host, database, user, port, password

    def connect(self):
        conn = None
        host, database, user, port,  password = self.config()
        try:
            # connect to the PostgreSQL server
            #print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(host=host, database=database,
                                    user=user, password=password, port=port)

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit(-1)
        return conn

