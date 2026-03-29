import numpy as np

import os, sys, contextlib
import importlib

# We'll import the heavy `renderer` module lazily inside `__init__` while
# temporarily suppressing stdout/stderr to avoid noisy C library messages
# from Panda3D/OpenAL/ALSA during initialization.
renderer = None

@contextlib.contextmanager
def suppress_output():
    fd_stdout = sys.stdout.fileno()
    fd_stderr = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(fd_stdout)
        old_stderr = os.dup(fd_stderr)
        try:
            os.dup2(devnull.fileno(), fd_stdout)
            os.dup2(devnull.fileno(), fd_stderr)
            yield
        finally:
            os.dup2(old_stdout, fd_stdout)
            os.dup2(old_stderr, fd_stderr)
            os.close(old_stdout)
            os.close(old_stderr)

import gymnasium as gym
import gymnasium

from gymnasium import spaces

import cv2

# Put pacsim build artifacts first so a stale local pacsim_pybind*.so does not shadow them.
sys.path.insert(0, "/root/pacsim_ws/build/pacsim")
import pacsim_pybind

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.use('Agg')

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import time

import math

from line_profiler import profile

import termcolor

class pacsimEnv(gymnasium.Env):
    def __init__(self, param=None):
        # print(param)
        print("Init, params {0}".format(param))
        self.useCamSim = False
        try:
            self.useCamSim = param["cam_sim"]
        except:
            self.useCamSim = True

        try:
            self.lambda_progress = param["lambda_progress"]
        except:
            self.lambda_progress = 0.02
        
        try:
            self.lambda_tracking = param["lambda_tracking"]
        except:
            self.lambda_tracking = 0.003

        try:
            self.lambda_finish = param["lambda_finish"]
        except:
            self.lambda_finish = 10.0

        try:
            self.lambda_collition = param["lambda_collition"]
        except:
            self.lambda_collition = 10.0

        try:
            self.lambda_stand = param["lambda_stand"]
        except:
            self.lambda_stand = 0.5

        try:
            self.lambda_constant = param["lambda_constant"]
        except:
            self.lambda_constant = 0.1

        try:
            self.lambda_slipAngle = param["lambda_slipAngle"]
        except:
            self.lambda_slipAngle = 0.005

        try:
            self.lambda_slipRatio = param["lambda_slipRatio"]
        except:
            self.lambda_slipRatio = 0.05

        try:
            self.lambda_actionRate = param["lambda_actionRate"]
        except:
            self.lambda_actionRate = 0.002

        try:
            self.lambda_lateral_consistency = param["lambda_lateral_consistency"]
        except:
            self.lambda_lateral_consistency = 0.001

        try:
            self.lambda_longitudinal_consistency = param["lambda_longitudinal_consistency"]
        except:
            self.lambda_longitudinal_consistency = 0.0002

        if(self.useCamSim):
            # Import and initialize the renderer while suppressing noisy output
            with suppress_output():
                try:
                    # Prefer setting Panda3D PRC options before ShowBase initialization
                    from panda3d.core import load_prc_file_data
                    load_prc_file_data('', 'window-type offscreen')
                    load_prc_file_data('', 'audio-library-name null')
                    load_prc_file_data('', 'notify-level warning')
                except Exception:
                    pass
                # import the renderer module lazily
                try:
                    renderer = importlib.import_module('renderer')
                except Exception:
                    # fallback to relative import if running as a package
                    renderer = importlib.import_module('.renderer', package=__package__)
                self.panda3dRenderer = renderer.Game()
            # step one frame to finish initialization
            self.panda3dRenderer.taskMgr.step()
        self.M = 8
        self.rangefinder_angles = np.linspace(-1.0, 1.0, 2*self.M+1)
        self.rangefinder_angles = np.power(np.abs(self.rangefinder_angles),1) * np.sign(self.rangefinder_angles)
        self.rangefinder_angles = self.rangefinder_angles * np.pi/2.0

        self.interval = 1.0/10.0
        self.pFL = np.array([1.65,0.72+0.11,0.0])
        self.pFR = np.array([1.65,-(0.72+0.11),0.0])
        self.pRL = np.array([-1.0,0.72+0.11,0.0])
        self.pRR = np.array([-1.0,-(0.72+0.11),0.0])

        self.useComplexModel = True
        self.outputRPM = True
        self.outputCurrentSteering = True
        self.outputLastActions = False
        self.outputIMU = True
        self.outputRangefinder = False

        self.minTorque = -22.0
        self.maxTorque = 22.0

        self.minSteering = -2.0
        self.maxSteering = 2.0
        
        self.maxRange = 75.0

        self.maxRpm = 20000.0

        self.maxSpeed = 40.0
        self.maxYawRate = 10.0

        self.maxImuAcceleration = 20.0
        
        self.emptyImageCode = np.zeros((3, 306, 256), dtype=np.uint8)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        rangeLower = np.zeros((self.M*2+1))
        rangeUpper = np.ones((self.M*2+1))
        lowerBound = np.concatenate((rangeLower,np.array([-1.0,-1.0,-1.0])))
        upperBound = np.concatenate((rangeUpper,np.array([1.0,1.0,1.0])))
        self.outputDim = self.M*2+1+3
        velLower = np.array([-1.0,-1.0])
        velUpper = np.array([1.0,1.0])
        rpmLower = None
        rpmUpper = None
        steerLower = None
        steerUpper = None
        imuLower = None
        imuUpper = None
        if(self.outputRPM):
            rpmLower = (-1000.0/self.maxRpm)*np.ones((4))
            rpmUpper = (21_000.0/self.maxRpm)*np.ones((4))
            lowerBound = np.concatenate((lowerBound,rpmLower))
            upperBound = np.concatenate((upperBound,rpmUpper))
            self.outputDim += 4
        if(self.outputCurrentSteering):
            steerLower = -1.0*np.ones((1))
            steerUpper = 1.0*np.ones((1))
            lowerBound = np.concatenate((lowerBound,steerLower))
            upperBound = np.concatenate((upperBound,steerUpper))
            self.outputDim += 1
        if(self.outputLastActions):
            lastActionLower = np.array([self.minSteering, self.minTorque, self.minTorque, self.minTorque, self.minTorque])
            lastActionUpper = np.array([self.maxSteering, self.maxTorque, self.maxTorque, self.maxTorque, self.maxTorque])
            lowerBound = np.concatenate((lowerBound,lastActionLower))
            upperBound = np.concatenate((upperBound,lastActionUpper))
            self.outputDim += 5
        if(self.outputIMU):
            imuLower = -1.0*np.ones((3))
            imuUpper = 1.0*np.ones((3))
            lowerBound = np.concatenate((lowerBound,imuLower))
            upperBound = np.concatenate((upperBound,imuUpper))
            self.outputDim += 2
        self.observation_space = spaces.Dict({
        "ranges": spaces.Box(low=rangeLower, high=rangeUpper, shape=(self.M*2+1,), dtype=np.float32),
        "velocity": spaces.Box(low=velLower, high=velUpper, shape=(2,), dtype=np.float32),
        "rpm": spaces.Box(low=rpmLower, high=rpmUpper, shape=(4,), dtype=np.float32),
        "steer": spaces.Box(low=steerLower, high=steerUpper, shape=(1,), dtype=np.float32),
        "imu": spaces.Box(low=imuLower, high=imuUpper, shape=(3,), dtype=np.float32),
        "cameraFront": spaces.Box(low=0, high=255, shape=(3, 306, 256,), dtype=np.uint8),
        "cameraLeft": spaces.Box(low=0, high=255, shape=(3, 306, 256,), dtype=np.uint8),
        "cameraRight": spaces.Box(low=0, high=255, shape=(3, 306, 256,), dtype=np.uint8),
        })
        self.trackNr = 0

        self.reset()

    @profile
    def _get_obs(self, ranges, velocity, rpm, steer, imu, cam):
        retDict = {
            "ranges": ranges,
            "velocity": velocity,
            "rpm": rpm,
            "steer": steer,
            "imu": imu,
        }
        if(self.outputRangefinder):
            retDict["ranges"] = ranges
        if(self.useCamSim):
            retDict["cameraLeft"] = cam[0].transpose(2,1,0)
            retDict["cameraFront"] = cam[1].transpose(2,1,0)
            retDict["cameraRight"] = cam[2].transpose(2,1,0)
        else:
            retDict["cameraLeft"] = self.emptyImageCode
            retDict["cameraFront"] = self.emptyImageCode
            retDict["cameraRight"] = self.emptyImageCode
        return retDict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([0.0,0.0,0.0])
        self.currentSteer = 0.0
        self.currentTorque = 0.0
        self.toruqes = [0.0,0.0,0.0,0.0]
        self.cameraImages = None
        self.rays = []
        self.trackNr = self.np_random.integers(0, 999999)
        if(self.useComplexModel):
            self.model = pacsim_pybind.VehicleModel4Wheel()
        else:
            self.model = pacsim_pybind.VehicleModel()
        c = pacsim_pybind.Config("/root/pacsim_ws/src/config/vehicleModel.yaml")
        c2 = c.getElement("vehicle_model")
        self.model.readConfig(c2)

        self.time = 0.0
        self.deadTime = 0.0
        try:
            self.deadTime = options["dead_time"]
        except:
            pass

        self.deadTimeSteering = pacsim_pybind.ScalarDeadtime(self.deadTime)
        self.deadTimeRPMSetpoints = pacsim_pybind.WheelsDeadtime(self.deadTime)
        self.deadTimeMaxTorques = pacsim_pybind.WheelsDeadtime(self.deadTime)
        self.deadTimeMinTorques = pacsim_pybind.WheelsDeadtime(self.deadTime)


        left_positions = []
        right_positions = []
        start_position = np.array([0,0,0])
        self.start_orientation = np.array([0,0,0])

        mapFiles = []
        mapFiles.append("/root/workspace/tracks/FSE22.yaml")
        mapFiles.append("/root/workspace/tracks/FSE22_test.yaml")
        mapFiles.append("/root/workspace/tracks/FSE23.yaml")
        mapFiles.append("/root/workspace/tracks/FSG19.yaml")
        mapFiles.append("/root/workspace/tracks/FSG21.yaml")
        mapFiles.append("/root/workspace/tracks/FSG23.yaml")
        mapFiles.append("/root/workspace/tracks/FSS19.yaml")
        mapFiles.append("/root/workspace/tracks/FSS22_V1.yaml")
        mapFiles.append("/root/workspace/tracks/FSS22_V2.yaml")
        mapFiles.append("/root/workspace/tracks/FSO20.yaml")

        mapFiles.append("/root/workspace/tracks/FSI24.yaml")
        mapFiles.append("/root/workspace/tracks/FSCZ24.yaml")

        try:
            mapFiles = options["map_files"]
        except:
            mapFiles = mapFiles

        mapFile = mapFiles[self.np_random.integers(0, len(mapFiles))]
        flipY = self.np_random.random() < 0.5
        self.flipX = self.np_random.random() < 0.5
        numStartPoints = 10
        quartile = self.np_random.integers(0, numStartPoints)

        try:
            if(options["noAugment"] == True):
                flipY = False
                self.flipX = False
                quartile = 0
        except:
            pass

        print("TrackNr {0}, mapFile {1}, flipY {2}, flipX {3}, quartile {4}".format(self.trackNr, mapFile, flipY, self.flipX, quartile))
        self.map = pacsim_pybind.loadMap(mapFile, start_position, self.start_orientation, flipY)
        self.left_lane = []
        self.right_lane = []
        for l in self.map.left_lane:
            a = l.position
            self.left_lane.append(a)
        for r in self.map.right_lane:
            a = r.position
            self.right_lane.append(a)
        self.path_left_point_indices = self.map.path_left_point_indices
        self.path_right_point_indices = self.map.path_right_point_indices

        points = []
        for i in range(0, len(self.path_left_point_indices)):
            p1 = self.map.left_lane[self.path_left_point_indices[i]].position
            p2 = self.map.right_lane[self.path_right_point_indices[i]].position
            p3 = 0.5*(p1+p2)
            points.append(p3)
        points.append(points[0])
        distances = [0]
        last = points[0]
        xs2 = [points[0][0]]
        ys2 = [points[0][1]]
        for i in points:
            dist = np.linalg.norm(i-last)
            if(dist > 2):
                last = i
                xs2.append(i[0])
                ys2.append(i[1])
                distances.append(dist+distances[-1])
        self.xs_spline = pacsim_pybind.CubicSpline(distances,xs2)
        self.ys_spline = pacsim_pybind.CubicSpline(distances,ys2)
        

        self.middleLineLength = distances[-1]
        segmentPoint = self.middleLineLength * quartile/numStartPoints

        
        self.startArc = segmentPoint
        self.arcLocalization = segmentPoint
        self.lastArc = segmentPoint
        endArcDist = 7.0
        if(self.flipX):
            endArcDist *= -1.0
        self.endArc = (self.startArc+endArcDist) % self.middleLineLength
        start_position = np.array([self.xs_spline(segmentPoint), self.ys_spline(segmentPoint), 0.0])
        start_angle = np.arctan2(self.ys_spline.derivative(segmentPoint), self.xs_spline.derivative(segmentPoint))
        if(self.flipX):
            self.start_orientation = np.array([self.start_orientation[0], self.start_orientation[1], self.start_orientation[2]+np.pi+start_angle])
        else:
            self.start_orientation = np.array([self.start_orientation[0], self.start_orientation[1], self.start_orientation[2]+start_angle])
        self.model.setOrientation(self.start_orientation)
        self.model.setPosition(start_position)
        self.orientation = self.start_orientation
        self.position = self.model.getPosition()


        self.trackNr += 1

        for l in self.map.left_lane:
            left_positions.append(l.position)
        for r in self.map.right_lane:
            right_positions.append(r.position)
        

        left_poly = Polygon(left_positions)
        right_poly = Polygon(right_positions)
        if(left_poly.area > right_poly.area):
            self.outer_poly = left_poly
            self.inner_poly = right_poly
        else:
            self.outer_poly = right_poly
            self.inner_poly = left_poly
        self.segsLeft = []
        for i in range(0,len(self.left_lane)):
            p1 = np.array(self.left_lane[i-1][0:2])
            p2 = np.array(self.left_lane[i][0:2])
            o = p1
            d = p2
            self.segsLeft.append((o,d))


        self.segsRight = []
        for i in range(0,len(self.right_lane)):
            p1 = np.array(self.right_lane[i-1][0:2])
            p2 = np.array(self.right_lane[i][0:2])
            o = p1
            d = p2
            self.segsRight.append((o,d))



        if(self.useCamSim):
            self.panda3dRenderer.removeCones()
            if(not self.flipX):
                self.panda3dRenderer.addBlueCones(left_positions)
                self.panda3dRenderer.addYellowCones(right_positions)
            else:
                self.panda3dRenderer.addBlueCones(right_positions)
                self.panda3dRenderer.addYellowCones(left_positions)
            image = self.renderFrames(self.position, self.orientation, self.model.getSteeringWheelAngle(), self.model.getWheelOrientations())
            self.cameraImages = image

        self.rewards = []

        self.lastAfterLine = False
        self.odometer = 0
        self.lapCount = 0
        self.frameCounter = 0
        ranges, self.rays = pacsim_pybind.runRangefinder(self.position, self.orientation, self.rangefinder_angles, self.segsLeft, self.segsRight)
        ranges = np.array(ranges)
        ranges = np.clip(ranges / self.maxRange,0.0,1.0)
        ret = self._get_obs(ranges, np.zeros(2), np.zeros(4), np.zeros(1), np.zeros(3), self.cameraImages)
        info  = {"laptime" : 0.0}

        self.lastActionArray = np.zeros(5)
        self.prevActionArray = np.zeros(5)
        return ret, info

    # @profile
    def renderFrames(self,position, orientation, steeringAngle, wheelOrientation):
        yaw = orientation[2]
        self.panda3dRenderer.updateCarPose(position[0], position[1], 180.0*yaw/np.pi)
        wheelAngles = [0.0,0.0,0.0,0.0]
        self.panda3dRenderer.updateSteering(180.0*steeringAngle/np.pi)
        self.panda3dRenderer.updateWheelRotations(*wheelAngles)
        self.panda3dRenderer.taskMgr.step()
        imageFront = self.panda3dRenderer.imbufclass.get_rgb_array()
        imageFront = np.ascontiguousarray(imageFront, dtype=np.uint8)

        imageLeft = self.panda3dRenderer.imbufclass2.get_rgb_array()
        imageLeft = np.ascontiguousarray(imageLeft, dtype=np.uint8)

        imageRight = self.panda3dRenderer.imbufclass3.get_rgb_array()
        imageRight = np.ascontiguousarray(imageRight, dtype=np.uint8)

        return imageLeft, imageFront, imageRight

    def rotMat2d(self, angle):
        ret = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return ret

    def render(self):
        fig, ax = plt.subplots()

        left_pos = []
        for i in self.map.left_lane:
            left_pos.append(np.array(i.position[0:2]))
        left_pos.append(np.array(self.map.left_lane[0].position[0:2]))
        left_pos = np.array(left_pos)

        right_pos = []
        for i in self.map.right_lane:
            right_pos.append(np.array(i.position[0:2]))
        right_pos.append(np.array(self.map.right_lane[0].position[0:2]))
        right_pos = np.array(right_pos)


        ax.plot(left_pos.T[0], left_pos.T[1], "b", linewidth=1.5, markersize=1.5)
        ax.plot(right_pos.T[0], right_pos.T[1], "y", linewidth=1.5, markersize=1.5)
        ax.plot(left_pos.T[0], left_pos.T[1], "ob", linewidth=1.5, markersize=1.5)
        ax.plot(right_pos.T[0], right_pos.T[1], "oy", linewidth=1.5, markersize=1.5)

        for i in self.rays:
            ax.plot([i[0][0],i[0][0]+i[1][0]*i[2]], [i[0][1],i[0][1]+i[1][1]*i[2]],"r", linewidth=1)

        ego_pos = self.position
        ax.plot([ego_pos[0]], [ego_pos[1]],"ok",markersize=1)

        pointsCar = []
        r = self.rotMat2d(self.orientation[2])
        pointsCar.append(np.array(self.position[0:2] + r @ self.pFL[0:2]))
        pointsCar.append(np.array(self.position[0:2] + r @ self.pRL[0:2]))
        pointsCar.append(np.array(self.position[0:2] + r @ self.pRR[0:2]))
        pointsCar.append(np.array(self.position[0:2] + r @ self.pFR[0:2]))
        pointsCar.append(pointsCar[0])
        pointsCar = np.array(pointsCar)
        ax.plot(pointsCar.T[0], pointsCar.T[1], "-k", markersize=1.0, linewidth=0.7)
            

        midXs = []
        midYs = []
        for i in range(0, len(self.path_left_point_indices)):
            p1 = self.map.left_lane[self.path_left_point_indices[i]].position
            p2 = self.map.right_lane[self.path_right_point_indices[i]].position
            p3 = 0.5*(p1+p2)
            midXs.append(p3[0])
            midYs.append(p3[1])

        x_splines = np.linspace(0, self.middleLineLength, 300)
        splineXs = []
        splineYs = []
        for i in x_splines:
            splineXs.append(self.xs_spline(i))
            splineYs.append(self.ys_spline(i))

        ax.set_aspect('equal')

        fig.canvas.draw()

        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        fig.clear()
        plt.close(fig)
        image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        timestring = 't=' + str(round(self.time,3))
        image = np.array(image, copy=True)
        image = cv2.putText(image, timestring, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        vel = self.model.getVelocity()[0:2]
        velocityString = "v=" + str([round(vel[0],2), round(vel[1],2)])
        image = cv2.putText(image, velocityString, (50,100), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        yawrate = self.model.getAngularVelocity()[2]
        yawrateString = "yawrate=" + str(round(yawrate,3))
        image = cv2.putText(image, yawrateString, (50,150), font, 
                fontScale, color, thickness, cv2.LINE_AA)

        
        trackNrString = "trackNumber=" + str(self.trackNr)
        image = cv2.putText(image, trackNrString, (500,50), font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        acc = self.model.getAcceleration()[0:2]
        accString = "acc=" + str([round(acc[0],2), round(acc[1],2)])
        image = cv2.putText(image, accString, (500,100), font, 
                        fontScale, color, thickness, cv2.LINE_AA)


        steeringString = "steering=" + str(round(self.currentSteer,3))
        image = cv2.putText(image, steeringString, (1000,50), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        flString = "torque FL=" + str(round(self.toruqes[0],2))
        frString = "torque FR=" + str(round(self.toruqes[1],2))
        image = cv2.putText(image, flString, (1000,100), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, frString, (1320,100), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        rlString = "torque RL=" + str(round(self.toruqes[2],2))
        rrString = "torque RR=" + str(round(self.toruqes[3],2))
        image = cv2.putText(image, rlString, (1000,150), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, rrString, (1320,150), font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def getReward(self, finished, collided):


        episode_rew = 0
        curVel = self.model.getVelocity()
        omega = self.model.getAngularVelocity()

        def cross(a,b):
            ret = np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])
            return ret
        
        vFL = curVel + cross(omega, self.pFL)
        vFR = curVel + cross(omega, self.pFR)
        vRL = curVel + cross(omega, self.pRL)
        vRR = curVel + cross(omega, self.pRR)

        alphaFL = 0.0
        alphaFR = 0.0
        alphaRL = 0.0
        alphaRR = 0.0
        if(np.abs(curVel[0] > 2)):
            if(np.abs(vFL[0] > 0.1)):
                alphaFL = np.abs(np.arctan2(vFL[1], vFL[0]) - self.currentSteer)
            if(np.abs(vFR[0] > 0.1)):
                alphaFR = np.abs(np.arctan2(vFR[1], vFR[0]) - self.currentSteer)
            if(np.abs(vRL[0] > 0.1)):
                alphaRL = np.abs(np.arctan2(vRL[1], vRL[0]))
            if(np.abs(vRR[0] > 0.1)):
                alphaRR = np.abs(np.arctan2(vRR[1], vRR[0]))
        # print("Alpha FL: {0}, FR: {1}, RL: {2}, RR: {3}".format(alphaFL, alphaFR, alphaRL, alphaRR))


        kappaFL = 0.0
        kappaFR = 0.0
        kappaRL = 0.0
        kappaRR = 0.0
        gearRatio = 12.23
        wheelRadius = 0.206
        rpm2ms = wheelRadius * 2.0 * np.pi / (gearRatio * 60.0)
        def getSlip(wheelspeed, vx):
            eps = 0.0001
            ret = np.abs(wheelspeed-vx) / max(np.abs(vx),eps)
            return ret

        if(np.abs(curVel[0] > 2)):
            kappaFL = getSlip(self.wheelspeeds.FL*rpm2ms, vFL[0])
            kappaFR = getSlip(self.wheelspeeds.FR*rpm2ms, vFR[0])
            kappaRL = getSlip(self.wheelspeeds.RL*rpm2ms, vRL[0])
            kappaRR = getSlip(self.wheelspeeds.RR*rpm2ms, vRR[0])

        # print("Long slip FL: {0}, FR: {1}, RL: {2}, RR: {3}".format(kappaFL, kappaFR, kappaRL, kappaRR))

        kappa = np.max([np.abs(kappaFL), np.abs(kappaFR), np.abs(kappaRL), np.abs(kappaRR)])
        if(not self.useComplexModel):
            kappa = 0.0

        alpha = max(min(alphaFL, alphaFR), min(alphaRL, alphaRR))
        alpha = np.rad2deg(alpha)
        # punish standstill
        if(curVel[0] >= 2.0):
            episode_rew = 0.0*10.0
        else:
            episode_rew = -50.0
        # add penalty for each timestep to incentivise shorter episodes (faster driving)
        episode_rew += -10.0

        def signed_modulo_distance(a, b, m):
            return ((b - a + m/2) % m) - m/2

        sdot = signed_modulo_distance(self.lastArc, self.arcLocalization, self.middleLineLength) / self.interval

        if(self.flipX):
            sdot = -1.0 * sdot

        actionDelta = self.lastActionArray - self.prevActionArray
        actionRateReg = (5*np.abs(actionDelta[0]/self.maxSteering) + np.abs(actionDelta[1]/self.maxTorque) + np.abs(actionDelta[2]/self.maxTorque) + np.abs(actionDelta[3]/self.maxTorque) + np.abs(actionDelta[4]/self.maxTorque)) / self.interval
        
        r_progress = sdot
        r_tracking = -np.abs(self.curvCoords[1])
        r_finish = 1.0 if finished else 0.0
        r_collition = -1.0 if collided else 0.0
        r_stand = -1.0 if (curVel[0] < 2.0) else 0.0
        r_constant = -1.0
        r_slipAngle = -np.abs(alpha)
        r_slipRatio = -kappa
        r_actionRate = -actionRateReg
        r_lateral_consistency = -max(0,-((self.lastActionArray[1] - self.lastActionArray[2]) * (self.lastActionArray[3] - self.lastActionArray[4])))
        r_longitudinal_consistency = -max(0,-((self.lastActionArray[1] + self.lastActionArray[2]) * (self.lastActionArray[3] + self.lastActionArray[4])))

        total_reward = self.lambda_progress * r_progress + self.lambda_tracking * r_tracking + self.lambda_finish * r_finish + self.lambda_collition * r_collition + self.lambda_stand * r_stand + self.lambda_constant * r_constant + self.lambda_slipAngle * r_slipAngle + self.lambda_slipRatio * r_slipRatio + self.lambda_actionRate * r_actionRate + self.lambda_lateral_consistency * r_lateral_consistency + self.lambda_longitudinal_consistency * r_longitudinal_consistency

        return total_reward

    @profile
    def step(self, actions):
        dt = 1.0/1000.0

        wmaxTorque = pacsim_pybind.Wheels()
        wminTorque = pacsim_pybind.Wheels()

        curVel = np.linalg.norm(self.model.getVelocity())
        wmaxRPM = pacsim_pybind.Wheels()

        self.toruqes = np.array([self.maxTorque*actions[1], self.maxTorque*actions[2], self.maxTorque*actions[3], self.maxTorque*actions[4]])
        if(self.useComplexModel):
            if(self.toruqes[0] >= 0):
                wmaxRPM.FL = 20000.0
                wmaxTorque.FL = self.toruqes[0]
                wminTorque.FL = 0.0
            else:
                wmaxRPM.FL = 0.0
                wmaxTorque.FL = 0.1
                wminTorque.FL = self.toruqes[0]

            if(self.toruqes[1] >= 0):
                wmaxRPM.FR = 20000.0
                wmaxTorque.FR = self.toruqes[1]
                wminTorque.FR = 0.0
            else:
                wmaxRPM.FR = 0.0
                wmaxTorque.FR = 0.1
                wminTorque.FR = self.toruqes[1]

            if(self.toruqes[2] >= 0):
                wmaxRPM.RL = 20000.0
                wmaxTorque.RL = self.toruqes[2]
                wminTorque.RL = 0.0
            else:
                wmaxRPM.RL = 0.0
                wmaxTorque.RL = 0.1
                wminTorque.RL = self.toruqes[2]

            if(self.toruqes[3] >= 0):
                wmaxRPM.RR = 20000.0
                wmaxTorque.RR = self.toruqes[3]
                wminTorque.RR = 0.0
            else:
                wmaxRPM.RR = 0.0
                wmaxTorque.RR = 0.1
                wminTorque.RR = self.toruqes[3]
        else:
            wmaxTorque.FL = self.toruqes[0]
            wmaxTorque.FR = self.toruqes[1]
            wmaxTorque.RR = self.toruqes[2]
            wmaxTorque.RR = self.toruqes[3]

        steering_action = self.maxSteering*actions[0]
        self.currentSteer = steering_action
        self.currentTorque = self.maxTorque*actions[1]

        self.deadTimeSteering.addVal(steering_action, self.time)
        self.deadTimeRPMSetpoints.addVal(wmaxRPM, self.time)
        self.deadTimeMaxTorques.addVal(wmaxTorque, self.time)
        self.deadTimeMinTorques.addVal(wminTorque, self.time)
        
        futureTime = self.interval * (self.frameCounter+1.0)
        wFric = pacsim_pybind.Wheels()
        wFric.FL = 1.0
        wFric.FR = 1.0
        wFric.RL = 1.0
        wFric.RR = 1.0

        while self.time < futureTime:
            if(self.deadTimeSteering.availableDeadTime(self.time)):
                steer = self.deadTimeSteering.getOldest()
                self.model.setSteeringSetpointFront(steer)
            if(self.deadTimeRPMSetpoints.availableDeadTime(self.time)):
                steer = self.deadTimeRPMSetpoints.getOldest()
                self.model.setRpmSetpoints(steer)
            if(self.deadTimeMaxTorques.availableDeadTime(self.time)):
                steer = self.deadTimeMaxTorques.getOldest()
                self.model.setMaxTorques(steer)
            if(self.deadTimeMinTorques.availableDeadTime(self.time)):
                steer = self.deadTimeMinTorques.getOldest()
                self.model.setMinTorques(steer)

            self.model.forwardIntegrate(dt, wFric)

            self.time += dt
        position = self.model.getPosition()
        orientation = self.model.getOrientation()
        currentSteer = self.model.getSteeringWheelAngle()
        self.position = position
        self.orientation = orientation
        pose = np.array([position[0], position[1], orientation[2]])
        curVel = np.linalg.norm(self.model.getVelocity())
        self.odometer += curVel * self.interval

        ranges, self.rays = pacsim_pybind.runRangefinder(self.position, self.orientation, self.rangefinder_angles, self.segsLeft, self.segsRight)
        ranges = np.array(ranges)
        ranges = np.clip(ranges / self.maxRange,0.0,1.0)

        if(self.useCamSim):
            images = self.renderFrames(position, orientation, self.model.getSteeringWheelAngle(), self.model.getWheelOrientations())
            self.cameraImages = images
        self.frameCounter += 1

        vel = self.model.getVelocity()
        rot = self.model.getAngularVelocity()
        acceleration = self.model.getAcceleration()
        self.acceleration = acceleration[0:2]
        velArray = np.array([vel[0],vel[1],rot[2]])
        rpms = self.model.getWheelspeeds()
        curTorques = self.model.getTorques()
        self.wheelspeeds = rpms
        rpmArray = np.array([rpms.FL, rpms.FR, rpms.RL, rpms.RR])
        self.prevActionArray = self.lastActionArray
        self.lastActionArray = np.array([steering_action, self.toruqes[0], self.toruqes[1], self.toruqes[2], self.toruqes[3]])

        velArrayNorm = np.array([vel[0]/self.maxSpeed,vel[1]/self.maxSpeed])
        rpmArrayOut = rpmArray / self.maxRpm
        steerNormArray  = np.array([np.clip(currentSteer/self.maxSteering,-1.0,1.0)])

        imuDataNorm = np.array([acceleration[0]/self.maxImuAcceleration, acceleration[1]/self.maxImuAcceleration, rot[2]/self.maxYawRate])
        imuNormArray = np.array(np.clip(imuDataNorm,-1.0,1.0))

        ret = self._get_obs(ranges, velArrayNorm, rpmArrayOut, steerNormArray, imuNormArray, self.cameraImages)
        info  = {"laptime" : 0.0}

        self.curvCoords = pacsim_pybind.findCurvlinearCoords(self.xs_spline, self.ys_spline, self.middleLineLength, self.position[0], self.position[1])
        self.arcLocalization = self.curvCoords[0]
        
        points = []
        r = self.rotMat2d(self.orientation[2])
        points.append(np.array([self.position[0:2] + r @ self.pFL[0:2]]).T)
        points.append(np.array([self.position[0:2] + r @ self.pFR[0:2]]).T)
        points.append(np.array([self.position[0:2] + r @ self.pRL[0:2]]).T)
        points.append(np.array([self.position[0:2] + r @ self.pRR[0:2]]).T)

        pointInTrack = []
        for i in points:
            inner = self.inner_poly.contains(Point(i))
            outer = self.outer_poly.contains(Point(i))
            res = outer and (not inner)
            pointInTrack.append(res)
        collided = not all(pointInTrack)

        def is_modulo_greater(a, b, mod):
            """
            Returns True if b comes after a in modulo `mod` space.
            """
            return (b - a) % mod < mod // 2


        afterLine = is_modulo_greater(self.endArc, self.arcLocalization, self.middleLineLength)
        if(self.flipX):
            afterLine = not afterLine
        crossedLine = (self.odometer > 20.0) and afterLine and (not self.lastAfterLine)
        if(afterLine and (self.odometer <= 20.0) and (not self.lastAfterLine)):
            self.timingStart = self.time
        reward = self.getReward(crossedLine, collided)
        self.lastAfterLine = afterLine
        self.lastArc = self.arcLocalization

        timeout = self.time > 120.0 or ((self.odometer < 1.0) and (self.time > 5.0)) or ((self.odometer < 3.0) and (self.time > 10.0)) or ((self.odometer < 10.0) and (self.time > 30.0)) or ((self.odometer < 20.0) and (self.time > 60.0))
        terminated = collided or crossedLine or timeout
        if((self.frameCounter % 100) == 3):
            actionDenormalized = np.array([self.maxSteering*actions[0], self.maxTorque*actions[1], self.maxTorque*actions[2], self.maxTorque*actions[3], self.maxTorque*actions[4]])
            # print("Time {0}, action: {1}, pose: {2}, velocity: {3}".format(self.time, actionDenormalized, pose, velArray))
            print("Time {0:.3f}, action: {1}, pose: {2}, velocity: {3}".format(
                self.time,
                np.array2string(actionDenormalized, precision=2, suppress_small=True),
                np.array2string(pose, precision=2, suppress_small=True),
                np.array2string(velArray, precision=2, suppress_small=True)
            ))
            # print("Cur torques FL {0}, FR: {1}, RL: {2}, RR: {3}".format(curTorques.FL, curTorques.FR, curTorques.RL, curTorques.RR))
        if(terminated):
            def b2s(arg, col="red"):
                if(arg):
                    return termcolor.colored("True", col)
                else:
                    return "False"
            print("Time {0}, Collided {1}, CrossedLine {2}, Timeout {3}, Pose {4}, Velocity {5}".format(self.time, b2s(collided), b2s(crossedLine, col="green"), b2s(timeout), np.array2string(pose, precision=2, suppress_small=True), np.array2string(velArray, precision=3, suppress_small=True)))
        if(crossedLine):
            info["laptime"] = self.time - self.timingStart
        if(timeout):
            print("Timeout")
        truncated = False
        return ret, reward, terminated, truncated, info