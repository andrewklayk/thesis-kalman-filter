import pandas as pd

gps = pd.read_csv('../inputs/gps.txt', sep=" ")
gps.drop(columns=['fix_qual', 'sats', 'VDOP', 'HDOP'], inplace=True)
imu = pd.read_csv('../inputs/imu.txt', sep=" ")
imu.drop(columns=['accelTimestamp', 'gyroTimestamp', 'gyroY', 'gyroZ'], inplace=True)
sensors = pd.read_csv('../inputs/sensors.txt', sep=" ")
sensorsRelevant = sensors[['systemTimestamp', 'WheelSpeedRearRight', 'WheelSpeedRearLeft']].copy()

resultData = pd.merge(gps, imu, how="outer", on=["systemTimestamp"])
resultData = pd.merge(resultData, sensorsRelevant, how="outer", on=["systemTimestamp"])
resultData.sort_values('systemTimestamp', inplace=True)
resultData.to_csv('inputs/total.csv', index=False)
