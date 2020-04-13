import sqlite3
import os


def query_func():
    # 500
    # START_RUN_ID = 2908
    # STOP_RUN_ID = 2919
    # 1000
    START_RUN_ID = 2908
    STOP_RUN_ID = 2926

    START_EPOCH = 0
    STOP_EPOCH = 800
    INTERVAL = 10
    BATCH_SIZE = 1000

    f = open("epoch_query.txt", "r")
    query_text = f.read()
    f.close()

    f = open("epoch_query_all_intervals.txt", "w")
    total_query = ""
    for start_epoch in range(START_EPOCH, STOP_EPOCH, INTERVAL):
        main_query = query_text
        main_query = main_query.replace("START_RUN_ID", str(START_RUN_ID))
        main_query = main_query.replace("STOP_RUN_ID", str(STOP_RUN_ID))
        main_query = main_query.replace("START_EPOCH", str(start_epoch))
        main_query = main_query.replace("STOP_EPOCH", str(start_epoch + INTERVAL))
        main_query = main_query.replace("BATCH_SIZE", str(BATCH_SIZE))
        interval_query = "SELECT * FROM\n"
        interval_query += "(\n"
        interval_query += main_query
        interval_query += "\n)"
        total_query += interval_query
        total_query += "\nUNION ALL\n"
    f.write(total_query)
    f.close()
    print("X")


query_func()