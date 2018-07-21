
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self,vehicle_mass,fuel_capacity,brake_deadband,
        decel_limit,accel_limit,wheel_radius,wheel_base,steer_radio,max_lat_accel,max_steer_angle):


        # TODO: Implement
        self.yaw_controller = YawController(wheel_base,steer_radio, 0.1,max_lat_accel,max_steer_angle)

        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0. # minimun throttle value
        mx = 0.5  # maximun trottle value
        self.throttle_controller = PID(kp,ki,kd,mn,mx)

        tau = 0.5  # 1(2pi tau), cutoff frecuency
        ts = .02 # sample time
        self.vel_lpf = LowPassFilter(tau,ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit  = accel_limit
        self.wheel_radius = wheel_radius
        self.last_time = rospy.get_time()
	'''
        #AVG Filter
        self.steeringBuffer = []
	self.nAvg=10

	'''
        self.str_lpf = LowPassFilter(0.2,0.1)

        

    def control(self, current_vel,dbw_enabled,linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
        current_vel = self.vel_lpf.filt(current_vel)
        steering = self.yaw_controller.get_steering(linear_vel,angular_vel,current_vel)
        steering = self.str_lpf.filt(steering)
        '''
	#AVG Filter
	self.steeringBuffer.append(steering)
	
	if len(self.steeringBuffer) > self.nAvg:
		self.steeringBuffer.pop(0)
	steering = sum(self.steeringBuffer)/float(len(self.steeringBuffer))
        '''
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time -  self.last_time
        self.last_time = current_time

        throttle =  self.throttle_controller.step(vel_error,sample_time)
        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400 # Nm
        elif throttle < 0.1 and vel_error < 0 :
            throttle = 0
            decel = max(vel_error,self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        return throttle,brake,steering

        
