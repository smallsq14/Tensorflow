import mysql.connector
from mysql.connector import errorcode

config_local = {
    'user': 'datauser',
    'password': '',
    'host': '127.0.0.1',
    'database': 'tensorflow',
    'raise_on_warnings': True
}

config_remote = {
    'user': 'datauser',
    'password': '',
    'host': '73.85.90.204',
    'port':7779,
    'database': 'tensorflow',
    'raise_on_warnings': True
}

def execute_command(command_sql,command_data):
    try:
        cnx = mysql.connector.connect(**config_remote)
        cursor = cnx.cursor()

        cursor.execute(command_sql, command_data)
        cnx.commit()
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Bad username or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    finally:
        cursor.close()
        cnx.close()


def open_connection():
    try:
       cnx = mysql.connector.connect(**config_remote)

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


def insert_cnn_run(name, imbalance, positive_negative, train_negative, train_positive, true_negative,false_positive,false_negative,true_positive,accuracy,incorrect,correct,notes,train_negative_imbalanced,train_positive_imbalanced,train_resampled_size):
    data_cnn_run = {'name':name,
                    'imbalance':name,
                    'positive_negative':positive_negative,
                    'train_negative':train_negative,
                    'train_positive':train_positive,
                    'true_negative':true_negative,
                    'false_positive':false_positive,
                    'false_negative':false_negative,
                    'true_positive':false_negative,
                    'accuracy':accuracy,
                    'incorrect':incorrect,
                    'correct':correct,
                    'notes':notes,
                    'train_negative_imbalanced':train_negative_imbalanced,
                    'train_positive_imbalanced':train_positive_imbalanced,
                    'train_resampled_size':train_resampled_size
                    }
    add_cnn_run_sql = ("INSERT INTO cnn_runs (name,imbalance,positive_negative,train_negative,train_positive,true_negative,false_positive,false_negative,true_positive,accuracy,incorrect,correct,notes,train_negative_imbalanced,train_positive_imbalanced,train_resampled_size) "
                        "VALUES (%(name)s, %(imbalance)s, %(positive_negative)s, %(train_negative)s, %(train_positive)s, %(true_negative)s, %(false_positive)s, %(false_negative)s, %(true_positive)s, %(true_positive)s, %(accuracy)s, %(incorrect)s, %(correct)s, %(notes)s, %(train_negative_imbalanced)s, %(train_positive_imbalanced)s %(train_resampled_size)s )")
    cursor.execute(add_cnn_run_sql, data_cnn_run)
    cnx.commit()

def insert_run_group(name, timestamp, notes, activeflag):
        data_run_group = {
            'name': name,
            'created_date': timestamp,
            'notes': notes,
            'active_flag': activeflag
        }
        add_run_group_sql = ("INSERT INTO run_groups "
                             "(name, created_date, notes, active_flag) "
                             "VALUES (%(name)s, %(created_date)s, %(notes)s, %(active_flag)s)")

        execute_command("",add_run_group_sql, data_run_group)


def update_run_group(id, name, timestamp, notes, activeflag):
    data_run_group = {
        'id': id,
        'name': name,
        'created_date': timestamp,
        'notes': notes,
        'active_flag': activeflag
    }
    add_run_group_sql = ("UPDATE run_groups "
                         "SET name = %(name)s, notes=%(notes)s, active_flag=%(active_flag)s where id=%(id)s ")

    execute_command(add_run_group_sql, data_run_group)

def get_active_run():
    query = ("Select * from run_group_runs where end_time is null and active_flag = 1")
    #execute_command()

def get_queued_runs():
    query = ("Select * from run_group_runs where end_time is null and active_flag = 0")


def testSql():

    insert_run_group('test1', '2016-05-05', 'test notes', 1)



if __name__ == '__main__':
    testSql()