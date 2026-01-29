# For each of the orbits in the yaml, compute the following:
# a, e, i, RAAN, aop, nu
# Orbit period (TP)
# Apogee and perigee radii, r-sub-a and r-sub-p respectively
#
# a - semi-major axis
# e - Eccentricity
# i - Inclination
# RAAN - Right Ascension of Ascending Node
# aop = Argument of Periapsis
# nu - True Anomaly
# TP - Orbit Period
# r-sub-a - Apoapsis radii
# r-sub-p - Perigee radii
#
# Assume Mu is 3.986004418 × 10^14 m^3/sec^2
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

        self.mu = 398600441800000

        self.r_vector = np.array([self.initial_x_pos, self.initial_y_pos, self.initial_z_pos])
        self.r_dot_vector = np.array([self.initial_x_vel, self.initial_y_vel, self.initial_z_vel])

        self.h_vector = self.determine_h(self.r_vector, self.r_dot_vector)

        z_hat = [0, 0, 1]

        self.inclination = self.determine_inclination(self.h_vector, z_hat)

        self.n_hat = self.determine_n_hat(z_hat, self.h_vector)
        
        self.raan = self.determine_right_ascension_of_ascending_node(self.n_hat[1], self.n_hat[0])

        self.b_vector = self.determine_b(self.r_vector, self.r_dot_vector, self.h_vector, self.mu)
        
        self.eccentricity = self.determine_eccentricity(self.b_vector, self.mu)

        self.energy = self.determing_energy(self.r_vector, self.r_dot_vector, self.mu)
        
        self.acceleration = self.determine_acceleration(self.energy, self.mu)
        
        self.orbital_period = self.determine_orbital_period(self.acceleration, self.mu)
        self.tp = self.orbital_period

        self.apogee_radii = self.determine_apogee_radii(self.acceleration, self.energy)

        self.perigee_radii = self.determine_perigee_radii(self.acceleration, self.energy)

        self.aop = self.determine_argument_of_periapsis(self.h_vector, self.n_hat, self.b_vector)

        self.eccentricity_vector = self.determine_eccentricity_vector(self.r_vector, self.r_dot_vector, self.mu)

        self.nu = self.determine_true_anomaly(self.r_vector, self.b_vector)


    def determine_acceleration(self, energy, mu):
        return -(mu/(2*energy))


    def determine_eccentricity(self, b: np.array, mu):
        return np.linalg.norm(b/mu)

    # Ref: https://www.youtube.com/watch?v=ENXHl7W8Iw0
    def determine_eccentricity_vector(self, r_vector, r_dot_vector, mu):
        return ((math.pow(np.linalg.norm(r_dot_vector), 2)/mu)-(1/np.linalg.norm(r_vector))) * r_vector - ((np.dot(r_vector, r_dot_vector))/mu) * r_dot_vector


    def determine_inclination(self, h_hat: np.array, z_hat: list):
        '''Returns in radians, convert to degrees if you need'''
        return math.acos(np.dot(h_hat, z_hat)/(np.linalg.norm(h_hat)))


    def determine_right_ascension_of_ascending_node(self, y, x):
        '''Returns in radians, convert to degrees if you need'''
        r = math.atan2(y, x)

        # Correct for if we are in quadrant 3 or 4 
        if r < 0:
            r = r + (2 * math.pi)

        return r

    def determine_true_anomaly(self, r_vector, b_vector):
        '''Returns in radians, convert to degrees if you need'''
        r = math.acos((np.dot(r_vector, b_vector))/(np.linalg.norm(r_vector) * np.linalg.norm(b_vector)))

        # Correct for if we are in quadrant 3 or 4
        if np.dot(r_vector, b_vector) < 0:
            r = (2 * math.pi) - r
        return r        


    def determine_argument_of_periapsis(self, h, n_hat, b):
        '''Returns in radians, convert to degrees if you need'''
        return math.atan2(np.dot(h/np.linalg.norm(h), np.cross(n_hat, b/np.linalg.norm(b))), np.dot(n_hat, b/np.linalg.norm(b)))


    def determine_orbital_period(self, a, mu):
        return 2 * math.pi * math.sqrt((math.pow(a, 3))/mu)


    def determine_apogee_radii(self, a, e):
        return a * (1 + e)


    def determine_perigee_radii(self, a, e):
        return a * (1 - e)


    def determing_energy(self, r, r_dot, mu):
        return (math.pow(np.linalg.norm(r_dot), 2)/2) - (mu/np.linalg.norm(r))


    def determine_h(self, r: np.array, r_dot: np.array):
        '''Returns the H-Hat, the cross product of the position vector (r) and the velocity vector (r-dot)'''
        return np.cross(r, r_dot)


    def determine_n_hat(self, z_hat: list, h: np.array):
        '''Returns the N-Hat'''
        return np.cross(z_hat, h)/np.linalg.norm(np.cross(z_hat, h))


    def determine_b(self, r, r_dot, h, mu):
        return np.cross(r_dot, h) - (mu * (r/np.linalg.norm(r))) 
    

def main():

    vectors_file = 'vectors.yaml'
    vector_data = read_in_yaml(vectors_file)

    for i in range(1,3):
        r_dot = np.array([vector_data['vectors'][f'vector{i}']['x_velocity'], vector_data['vectors'][f'vector{i}']['y_velocity'], vector_data['vectors'][f'vector{i}']['z_velocity']])

        ke = KeplerianElements(vector_data['vectors'][f'vector{i}']['x_pos'],
                               vector_data['vectors'][f'vector{i}']['y_pos'],
                               vector_data['vectors'][f'vector{i}']['z_pos'],
                               vector_data['vectors'][f'vector{i}']['x_velocity'],
                               vector_data['vectors'][f'vector{i}']['y_velocity'],
                               vector_data['vectors'][f'vector{i}']['z_velocity'])

        print(f'----- Vector {i} -----')
        print(f'Position Vector       : {ke.r_vector}')
        print(f'Velocity Vector       : {ke.r_dot_vector}')
        print(f'Acceleration          : {ke.acceleration} meters')
        print(f'Eccentricity          : {ke.eccentricity}')
        print(f'Inclination           : {math.degrees(ke.inclination)} Degrees')
        print(f'RAAN                  : {math.degrees(ke.raan)} Degress')
        print(f'Argument of Periapsis : {math.degrees(ke.aop)} Degrees')
        print(f'Nu                    : {math.degrees(ke.nu)} Degrees')
        print(f'Orbit Period          : {ke.tp} seconds')
        print(f'Apogee Radii          : {ke.apogee_radii} meters')
        print(f'Perigee Radii         : {ke.perigee_radii} meters')
        print()

if __name__ == '__main__':
    main()
