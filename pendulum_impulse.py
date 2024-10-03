from math import sqrt

GRAVITY = 9.8

def friction_force(friction_coefficient:float, normal_force:float)->float:
    return friction_coefficient * normal_force

def solve_car_initial_velocity(distance, weight, friction_coefficient):
    friction_energy = friction_force(friction_coefficient, weight) * distance
    return sqrt(2 * friction_energy / weight)

def solve_ball_final_velocity(coefficient_of_restitution, car_initial_velocity):
    return car_initial_velocity / coefficient_of_restitution

def calculate_coefficient_of_restitution(bounce_height:float, drop_height:float)->float:
    return sqrt(bounce_height / drop_height)

def calculate_pendulum_ball_height(ball_final_velocity:float)->float:
    return ball_final_velocity**2 / (2 * GRAVITY)

pendulum_length = 6
target_distance = 15
car_mass = 0.3
car_friction_coefficient = 0.6
ball_mass = 0.25
drop_height = 5
bounce_height = 4.5

car_v1 = solve_car_initial_velocity(target_distance, car_mass, car_friction_coefficient)
cor = calculate_coefficient_of_restitution(bounce_height, drop_height)
ball_v2 = solve_ball_final_velocity(cor, car_v1)
ball_height = calculate_pendulum_ball_height(ball_v2)
valid_height = ball_height < pendulum_length

print(f"Initial velocity of the car: {car_v1:.2f} m/s")
print(f"Coefficient of restitution: {cor:.2f}")
print(f"Final velocity of the ball: {ball_v2:.2f} m/s")
print(f"Height of the ball: {ball_height:.2f} m")
print(f"Is the ball height valid: {valid_height}")
