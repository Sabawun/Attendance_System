import mysql.connector
from mysql.connector import Error


def check_username_password():
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='Attendance',
                                             user='root',
                                             password='Pakistan1.')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            cursor = connection.cursor()
            id = input("Please enter user_id: ")
            password = input("Please enter password: ")
            cursor.execute('SELECT * FROM logIn WHERE id = %s AND password = %s', (id, password,))
            logIn = cursor.fetchone()
            if logIn:
                print("Found")
                cursor.close()
                connection.close()
            else:
                print("Wrong id or password. Try again")
                cursor.close()
                connection.close()
                return 0

    except Error as e:
        print("Error while connecting to MySQL", e)


check_username_password()
