import pandas as pd

gps = pd.read_csv('res/gps.txt', sep=" ")
gps.drop(columns=['fix_qual', 'sats', 'VDOP', 'HDOP'], inplace=True)
imu = pd.read_csv('res/imu.txt', sep=" ")
imu.drop(columns=['accelTimestamp', 'gyroTimestamp', 'gyroY', 'gyroZ'], inplace=True)
sensors = pd.read_csv('res/sensors.txt', sep=" ")
sensorsRelevant = sensors[['systemTimestamp', 'WheelSpeedRearRight', 'WheelSpeedRearLeft']].copy()

resultData = pd.merge(gps, imu, how="outer", on=["systemTimestamp"])
resultData = pd.merge(resultData, sensorsRelevant, how="outer", on=["systemTimestamp"])
resultData.sort_values('systemTimestamp', inplace=True)
resultData.to_csv('res/total.csv', index=False)
