import pyodbc


def courseList(student_id):
    server = 'tcp:34.65.134.105'
    database = 'Attendance'
    username = 'onur'
    password = 'onur12345'
    cnxn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=' + server + ';PORT=1433;DATABASE=' + database +
        ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    cursor.execute("SELECT CourseList FROM CoursePopulation WHERE StudentID = ?", student_id)
    LectureString = cursor.fetchone()[0]
    LList = []
    for lectures in LectureString.split(','):
        LList.append(lectures)
    LectureListx = []
    LectureSchedulesx = []
    # Courses are seperated. now Check each
    for i in range(len(LList)):
        cursor.execute("SELECT CourseName, CourseSchedule FROM CourseInfo WHERE CourseID = ?", LList[i])
        values = cursor.fetchone()
        LectureListx.append(values[0])
        LectureSchedulesx.append(values[1])
    return LectureListx, LectureSchedulesx
