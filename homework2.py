# Find E_0
# Find v_0
# Find M_0
# Find time of flight from perigee to v_0
# Find time of flight from v_0 to v = 65 degrees
# Verify that starting at v_0, compute the true anomaly after the time of flight above
# Starting at v_0, what is the true anomaly after 2700 seconds?
# Starting at v_0, what is the true anomaly after exactly two orbit periods
# What is the true anomaly after 15000 seconds?
#
# Compute the Keplarian elements of the given vector (Input as vector2 in the vectors.yaml)
# Find the perifocal position and velocity, r_vector_0 and v_vector_0
# Find f, g, f_dot, g_dot, for delta_v = 33 degrees
# Find r_vector and v_vector
#

import math
import yaml

import numpy as np

# Read in a yaml that has all the initial vectors for position and velocity
def read_in_yaml(file_name):
    with open(file_name, 'r') as f:
        data = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return data


class KeplerianElements():
    '''
    Generates the Keplerian Elements given the 6 required parameters:
    X-Position, X-Velocity
    Y-Position, Y-Velocity
    Z-Position, Z-Velocity

    Depends on numpy for finding dot and cross products 
    '''
    def __init__(self, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel):
        self.initial_x_pos = x_pos
        self.initial_x_vel = x_vel
        self.initial_y_pos = y_pos
        self.initial_y_vel = y_vel
        self.initial_z_pos = z_pos
        self.initial_z_vel = z_vel

        self.mu = 398600441800000 # From WGS84

        z_hat = [0, 0, 1]

        self.r_vector = np.array([self.initial_x_pos, self.initial_y_pos, self.initial_z_pos])
        self.r_dot_vector = np.array([self.initial_x_vel, self.initial_y_vel, self.initial_z_vel])

        self.h_vector = self.determine_h()
        self.angular_momentum_vector = self.h_vector

        self.inclination = self.determine_inclination(z_hat)

        self.n_hat = self.determine_n_hat(z_hat)
        
        self.raan = self.determine_right_ascension_of_ascending_node()

        self.b_vector = self.determine_b()
        
        self.eccentricity = self.determine_eccentricity()

        self.energy = self.determing_energy()
        
        self.acceleration = self.determine_acceleration()
        
        self.orbital_period = self.determine_orbital_period()
        self.tp = self.orbital_period

        self.apogee_radii = self.determine_apogee_radii()

        self.perigee_radii = self.determine_perigee_radii()

        self.aop = self.determine_argument_of_periapsis()

        self.nu = self.determine_true_anomaly()
        self.true_anomaly = self.nu

        self.eccentricity_anomaly = self.determine_eccentricity_anomaly()

        self.mean_anomaly = self.determine_mean_anomaly()
        
        self.mean_motion = self.determine_mean_motion()

        self.v0 = self.determine_v_0()

        self.E0 = self.determine_E_0()


    def determine_acceleration(self):
        return -(self.mu/(2 * self.energy))

    def determine_eccentricity(self):
        return np.linalg.norm(self.b_vector / self.mu)

    def determine_eccentricity_anomaly(self):
        n_e = np.dot(self.r_vector, self.r_dot_vector)/math.sqrt(self.mu*self.acceleration)
        d_e = 1 - (np.linalg.norm(self.r_vector)/self.acceleration)
        r = math.atan2(n_e, d_e)

        if np.dot(self.r_vector, self.b_vector) < 0:
            r = (2 * math.pi) + r

        return r
    
    def determine_arbitrary_eccentric_anomaly(self, angle):
        '''Takes in an angle in degrees'''
        radian_angle = math.radians(angle)
        return math.asin((math.sin(radian_angle * math.sqrt(1 - math.pow(self.eccentricity, 2))))/(1 - self.eccentricity * math.cos(radian_angle)))

    def determine_inclination(self, z_hat: list):
        '''Returns in radians, convert to degrees if you need'''
        return math.acos(np.dot(self.h_vector, z_hat)/(np.linalg.norm(self.h_vector)))

    def determine_right_ascension_of_ascending_node(self):
        '''Returns in radians, convert to degrees if you need'''
        r = math.atan2(self.n_hat[1], self.n_hat[0])

        # Correct for if we are in quadrant 3 or 4 
        if r < 0:
            r = r + (2 * math.pi)

        return r

    def determine_true_anomaly(self):
        '''Returns in radians, convert to degrees if you need'''
        r = math.acos((np.dot(self.r_vector, self.b_vector))/(np.linalg.norm(self.r_vector) * np.linalg.norm(self.b_vector)))

        # Correct for if we are in quadrant 3 or 4
        if np.dot(self.r_vector, self.b_vector) < 0:
            r = (2 * math.pi) - r
        return r        

    def determine_argument_of_periapsis(self):
        '''Returns in radians, convert to degrees if you need'''
        return math.atan2(np.dot(self.h_vector/np.linalg.norm(self.h_vector), np.cross(self.n_hat, self.b_vector/np.linalg.norm(self.b_vector))), np.dot(self.n_hat, self.b_vector/np.linalg.norm(self.b_vector)))

    def determine_orbital_period(self):
        return 2 * math.pi * math.sqrt((math.pow(self.acceleration, 3))/self.mu)

    def determine_apogee_radii(self):
        return self.acceleration * (1 + self.energy)

    def determine_perigee_radii(self):
        return self.acceleration * (1 - self.energy)

    def determing_energy(self):
        return (math.pow(np.linalg.norm(self.r_dot_vector), 2)/2) - (self.mu/np.linalg.norm(self.r_vector))

    def determine_h(self):
        '''Returns the H-Hat, the cross product of the position vector (r) and the velocity vector (r-dot)'''
        return np.cross(self.r_vector, self.r_dot_vector)

    def determine_n_hat(self, z_hat: list):
        '''Returns the N-Hat'''
        return np.cross(z_hat, self.h_vector)/np.linalg.norm(np.cross(z_hat, self.h_vector))

    def determine_b(self):
        return np.cross(self.r_dot_vector, self.h_vector) - (self.mu * (self.r_vector/np.linalg.norm(self.r_vector))) 

    def print_ke(self):
        print(f'Position Vector       : {self.r_vector}')
        print(f'Velocity Vector       : {self.r_dot_vector}')
        print(f'Acceleration          : {self.acceleration} meters')
        print(f'Eccentricity          : {self.eccentricity}')
        print(f'Inclination           : {math.degrees(self.inclination)} Degrees')
        print(f'RAAN                  : {math.degrees(self.raan)} Degress')
        print(f'Argument of Periapsis : {math.degrees(self.aop)} Degrees')
        print(f'Nu                    : {math.degrees(self.nu)} Degrees')
        print(f'Nu                    : {self.nu} Degrees')
        print(f'Orbit Period          : {self.tp} seconds')
        print(f'Apogee Radii          : {self.apogee_radii} meters')
        print(f'Perigee Radii         : {self.perigee_radii} meters')

    def determine_mean_motion(self):
        return math.sqrt(self.mu/math.pow(self.acceleration, 3))

    def determine_mean_anomaly(self):
        return self.eccentricity_anomaly - self.eccentricity * math.sin(self.eccentricity_anomaly)

    def determine_v_0(self):
        return self.nu

    def determine_E_0(self):
        return math.atan2(math.sin(self.eccentricity_anomaly), math.cos(self.eccentricity_anomaly))
    
    def determine_time_of_flight(self):
        return self.mean_anomaly/self.mean_motion
    
    def determine_time_to_angle(self, angle, perigee_passes=0):
        '''Provided an angle from 0-360, returns the seconds to reach that angle from Nu'''
        math.radians(angle)
        return math.sqrt(math.pow(self.acceleration, 3)/self.mu) * ((2 * math.pi * perigee_passes) + (self.eccentricity_anomaly - self.eccentricity * math.sin(self.eccentricity_anomaly)) - self.mean_anomaly)

    def determine_location_after_n_seconds(self, seconds):
        # This isn't quite right
        location = self.nu + (self.mean_motion * seconds)

        while location > (2 * math.pi):
            location = location - (2 * math.pi)
        
        return location



def main():

    vectors_file = 'vectors.yaml'
    vector_data = read_in_yaml(vectors_file)

    print(f'----- Vector 1 -----')
    ke1 = KeplerianElements(vector_data['vectors'][f'vector1']['x_pos'],
                               vector_data['vectors'][f'vector1']['y_pos'],
                               vector_data['vectors'][f'vector1']['z_pos'],
                               vector_data['vectors'][f'vector1']['x_velocity'],
                               vector_data['vectors'][f'vector1']['y_velocity'],
                               vector_data['vectors'][f'vector1']['z_velocity'])

    print(f'E_0                   : {ke1.E0} Radians')
    print(f'E_0                   : {math.degrees(ke1.E0)} Degrees')
    print(f'v_0                   : {ke1.v0} Radians')
    print(f'v_0                   : {math.degrees(ke1.v0)} Degrees')
    print(f'Mean Motion           : {ke1.mean_motion} radians')
    print(f'Mean Motion           : {math.degrees(ke1.mean_motion)} Degrees')
    print(f'Mean Anomaly          : {ke1.mean_anomaly}')
    print(f'Mean Anomaly          : {math.degrees(ke1.mean_anomaly)} Degrees')
    print(f'Time of Flight        : {ke1.determine_time_of_flight()} seconds')
    print(f'Eccentic angle at 65 degrees: {ke1.determine_arbitrary_eccentric_anomaly(65)}')
    print(f'Time to 65 Degrees    : {ke1.determine_time_to_angle(65)} seconds')
    print(f'Location after 2700s  : {math.degrees(ke1.determine_location_after_n_seconds(2700))} Degrees')
    print()

    exit()

    print(f'----- Vector 2 -----')
    ke2 = KeplerianElements(vector_data['vectors'][f'vector2']['x_pos'],
                               vector_data['vectors'][f'vector2']['y_pos'],
                               vector_data['vectors'][f'vector2']['z_pos'],
                               vector_data['vectors'][f'vector2']['x_velocity'],
                               vector_data['vectors'][f'vector2']['y_velocity'],
                               vector_data['vectors'][f'vector2']['z_velocity'])
    
    ke2.print_ke()
    

if __name__ == '__main__':
    main()
