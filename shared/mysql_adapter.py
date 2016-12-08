import mysql.connector
from mysql.connector import errorcode

config = {
  'user': 'datauser',
  'password': 'datauser',
  'host': '127.0.0.1',
  'database': 'tensorflow',
  'raise_on_warnings': True
}

def open_connection():
        try:
          cnx = mysql.connector.connect(**config)
        except mysql.connector.Error as err:
          if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
          elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
          else:
            print(err)
        else:
          cnx.close()

def close_connection():
        cnx.close()

def insert_cnn_results(name,imbalance,positive_negative,train_negative):


def insert_run_group(name,timestamp,notes,activeflag):
    data_run_group = {
	  'name':name,
	  'created_date':timestamp,
	  'notes':notes,
          'active_flag':activeflag
	}
    add_run_group_sql = ("INSERT INTO run_groups "
			     "(name, created_date, notes, active_flag) "
			     "VALUES (%(name)s, %(created_date)s, %(notes)s, %(active_flag)s)"

    cursor.execute(add_run_group_sql, data_run_group)
    cnx.commit()
        
open_connection()
insert_run_group('test1','2016-05-05','test notes',1)

