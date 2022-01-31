"""

"""

import numpy as np
import random
import gpxpy
import pandas as pd
#from geopy.distance import vincenty, geodesic
import matplotlib.pyplot as plt
import math
import geopy.distance
import utm
import decimal


xcart = []
ycart = []
z1cart = []
z2cart = []
xcartfirst = []
ycartfirst = []
xEstb1 = []
xEstb2 = []
xDrec1 = []
xDrec2 = []
xTrueg = []
yTrueg = []

"Import GPS Data"
with open('3296026.gpx') as fh:
    gpx_file = gpxpy.parse(fh)
    segment = gpx_file.tracks[0].segments[0]
    coords = pd.DataFrame([
    {'lat': p.latitude,
     'lon': p.longitude,
     'ele': p.elevation,
     } for p in segment.points])


"Compute delta between timestamps"
times = pd.Series([p.time for p in segment.points], name='time')
dt = np.diff(times.values) / np.timedelta64(1, 's')


def measuredistance(point1_x,point1_y,point2_x,point2_y):
    point1=(point1_x,point1_y)
    point2=(point2_x,point2_y)
    distance_points = geopy.distance.distance(point1,point2).meters
    return distance_points


def measuredistance2(point1_x,point1_y,point2_x,point2_y):
    x1, y1, z1, u = utm.from_latlon(point1_x,point1_y)
    x2, y2, z2, u = utm.from_latlon(point2_x,point2_y)
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist


def convertcartesiana(lat1,lon1):
    utm_conversion = utm.from_latlon(lat1,lon1)
    return utm_conversion


def calculate_angulars(lata,longa,latb,longb,timediff):
    X = math.cos(latb) * math.sin(longb-longa)
    Y = math.cos(lata) * math.sin(latb) - math.sin(lata) * math.cos(latb) * math.cos(longb -longa)
    bearing = math.atan2(X,Y)
    angularvelocity = bearing/timediff
    return angularvelocity


def angular_arctan(x1,y1,x2,y2,timed):
    theta1 = math.atan2(y1,x1)
    theta2 = math.atan2(y2,x2)
    anglevel = (theta2 - theta1) / timed
    return anglevel


# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([2, 2]) ** 2

show_animation = False


def observation(xTrue, xd, u, dtval,zgps):
    xTrue = motion_model(xTrue, u, dtval)

    # add noise to gps x-y
    z = zgps + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud, dtval)

    return xTrue, z, xd, ud


def motion_model(x, u, dtval):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[dtval * math.cos(x[2, 0]), 0],
                  [dtval * math.sin(x[2, 0]), 0],
                  [0.0, dtval],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u, dtval):
    """
    Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -dtval * v * math.sin(yaw), dtval * math.cos(yaw)],
        [0.0, 1.0, dtval * v * math.cos(yaw), dtval * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u, dtval):
    #  Predict
    xPred = motion_model(xEst, u, dtval)
    jF = jacob_f(xEst, u, dtval)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)
    zgps = np.zeros((2, 1))

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    for i in range(len(coords.lat) - 1):
        if (i > 0 and i <= 2000):
            utmconvert = convertcartesiana(coords.lat[i], coords.lon[i])
            utmconvertfirst = convertcartesiana(coords.lat[i-1], coords.lon[i-1])

            distance2p = measuredistance2(coords.lat[i-1], coords.lon[i-1], coords.lat[i], coords.lon[i])
            print("distance : " + str(distance2p))

            speed = distance2p / dt[i-1]
            print("measured speed : " + str(speed))

            angular_vel = angular_arctan(utmconvertfirst[0], utmconvertfirst[1], utmconvert[0], utmconvert[1],dt[i-1])

            u = np.array([[speed], [angular_vel]])

            zgps[0,0] = utmconvert[0]
            zgps[1,0] = utmconvert[1]

            if (i ==1):
                print(str(utmconvertfirst[0]) + " ," + str(utmconvertfirst[1] ))
                xTrue[0, 0] = utmconvertfirst[0]
                xTrue[1, 0] = utmconvertfirst[1]
                xTrue[2, 0] = math.atan2(utmconvert[1] - utmconvertfirst[1],utmconvert[0] - utmconvertfirst[0])
                xEst[0, 0]  = utmconvertfirst[0]
                xEst[1, 0]  = utmconvertfirst[1]
                xEst[2, 0]  = xTrue[2, 0]
                xcart.append(utmconvertfirst[0])
                ycart.append(utmconvertfirst[1])
                xTrueg.append(xTrue[0])
                yTrueg.append(xTrue[1])
                xEstb1.append(xEst[0])
                xEstb2.append(xEst[1])
                z1cart.append(utmconvertfirst[0])
                z2cart.append(utmconvertfirst[1])
            else:
                xTrue[2, 0] = math.atan2(utmconvert[1] - utmconvertfirst[1], utmconvert[0] - utmconvertfirst[0])



            xcart.append(utmconvert[0])
            ycart.append(utmconvert[1])

            xTrue, z, xDR, ud = observation(xTrue, xDR, u, dt[i-1],zgps)

            xTrueg.append(xTrue[0])
            yTrueg.append(xTrue[1])
            xDrec1.append(xDR[0])
            xDrec2.append(xDR[1])

            xEst, PEst = ekf_estimation(xEst, PEst, z, ud, dt[i-1])

            print(xTrue)
            print(xEst)

            z1cart.append(z[0])
            z2cart.append(z[1])
            xEstb1.append(xEst[0])
            xEstb2.append(xEst[1])

if __name__ == '__main__':
    main()


plt.plot(xTrueg,yTrueg)
plt.plot(z1cart,z2cart)
plt.plot(xEstb1,xEstb2)
plt.xlabel('y')
plt.ylabel('x')
plt.title(' Scatter Plot of gps points with noise')
plt.legend([" True trajectory","gps data" , "ekf"])
plt.show()




